import numpy
import random

from htm.bindings.sdr import SDR, Metrics
import htm.bindings.encoders
ScalarEncoder = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.algorithms import Classifier

def Within ( value, minimum, maximum, equality ):
# Checks if value is <= maximum and >= minimum.

    if equality:
        if value <= maximum and value >= minimum:
            return True
        else:
            return False
    else:
        if value < maximum and value > minimum:
            return True
        else:
            return False

class BallAgent:

    localDimX   = 100
    localDimY   = 100

    maxPredLocations = 40
    maxMemoryDist = 10

    def __init__( self, name, screenHeight, screenWidth, ballHeight, ballWidth, paddleHeight, paddleWidth ):

        self.ID = name

        self.screenHeight = screenHeight
        self.screenWidth  = screenWidth

        # Set up encoder parameters
        ballXEncodeParams    = ScalarEncoderParameters()
        ballYEncodeParams    = ScalarEncoderParameters()
        ballTEncodeParams    = ScalarEncoderParameters()
        centerXEncodeParams  = ScalarEncoderParameters()
        centerYEncodeParams  = ScalarEncoderParameters()
        paddleEncodeParams   = ScalarEncoderParameters()

        ballXEncodeParams.activeBits = 21
        ballXEncodeParams.radius     = 20
        ballXEncodeParams.clipInput  = False
        ballXEncodeParams.minimum    = -int( self.localDimX / 2 )
        ballXEncodeParams.maximum    = int( self.localDimX / 2 )
        ballXEncodeParams.periodic   = False

        ballYEncodeParams.activeBits = 21
        ballYEncodeParams.radius     = 20
        ballYEncodeParams.clipInput  = False
        ballYEncodeParams.minimum    = -int( self.localDimY / 2 )
        ballYEncodeParams.maximum    = int( self.localDimY / 2 )
        ballYEncodeParams.periodic   = False

        ballTEncodeParams.activeBits = 21
        ballTEncodeParams.radius     = 20
        ballTEncodeParams.clipInput  = False
        ballTEncodeParams.minimum    = 0
        ballTEncodeParams.maximum    = self.maxMemoryDist
        ballTEncodeParams.periodic   = False

        paddleEncodeParams.activeBits = 21
        paddleEncodeParams.radius     = 20
        paddleEncodeParams.clipInput  = False
        paddleEncodeParams.minimum    = -int( self.localDimY / 2 )
        paddleEncodeParams.maximum    = int( self.localDimY / 2 )
        paddleEncodeParams.periodic   = False

        centerXEncodeParams.activeBits = 5
        centerXEncodeParams.radius     = 20
        centerXEncodeParams.clipInput  = False
        centerXEncodeParams.minimum    = -int( screenWidth / 2 )
        centerXEncodeParams.maximum    = int( screenWidth / 2 )
        centerXEncodeParams.periodic   = False

        centerYEncodeParams.activeBits = 5
        centerYEncodeParams.radius     = 20
        centerYEncodeParams.clipInput  = False
        centerYEncodeParams.minimum    = -int( screenHeight / 2 )
        centerYEncodeParams.maximum    = int( screenHeight / 2 )
        centerYEncodeParams.periodic   = False

        # Set up encoders
        self.ballEncoderX    = ScalarEncoder( ballXEncodeParams )
        self.ballEncoderY    = ScalarEncoder( ballYEncodeParams )
        self.ballEncoderT    = ScalarEncoder( ballTEncodeParams )
        self.centerEncoderX  = ScalarEncoder( centerXEncodeParams )
        self.centerEncoderY  = ScalarEncoder( centerYEncodeParams )
        self.paddleEncoderY  = ScalarEncoder( paddleEncodeParams )

        self.encodingWidth = ( self.ballEncoderX.size + self.ballEncoderY.size + self.centerEncoderX.size +
            self.centerEncoderY.size + ( self.paddleEncoderY.size * 2 ) + self.ballEncoderT.size )

        self.sp = SpatialPooler(
            inputDimensions            = ( self.encodingWidth, ),
            columnDimensions           = ( 2048, ),
            potentialPct               = 0.85,
            potentialRadius            = self.encodingWidth,
            globalInhibition           = True,
            localAreaDensity           = 0,
            numActiveColumnsPerInhArea = 40,
            synPermInactiveDec         = 0.005,
            synPermActiveInc           = 0.04,
            synPermConnected           = 0.1,
            boostStrength              = 3.0,
            seed                       = -1,
            wrapAround                 = False
        )

        self.tp = TemporalMemory(
            columnDimensions          = ( 2048, ),
            cellsPerColumn            = 32,
            activationThreshold       = 16,
            initialPermanence         = 0.21,
            connectedPermanence       = 0.1,
            minThreshold              = 12,
            maxNewSynapseCount        = 20,
            permanenceIncrement       = 0.1,
            permanenceDecrement       = 0.1,
            predictedSegmentDecrement = 0.0,
            maxSegmentsPerCell        = 128,
            maxSynapsesPerSegment     = 32,
            seed                      = 42
        )

        self.ballLocalXClass    = Classifier( alpha = 1 )
        self.ballLocalYClass    = Classifier( alpha = 1 )
        self.ballGlobalXClass   = Classifier( alpha = 1 )
        self.ballGlobalYClass   = Classifier( alpha = 1 )
        self.ballGlobalTClass   = Classifier( alpha = 1 )

        self.paddleALocalYClass = Classifier( alpha = 1 )
        self.paddleBLocalYClass = Classifier( alpha = 1 )
        self.paddleAMotorClass  = Classifier( alpha = 1 )
        self.paddleBMotorClass  = Classifier( alpha = 1 )

        self.predPositions = []
        self.memBuffer = [ [ 0, 0, 0, 0, 1, 1 ] ] * self.maxMemoryDist

    def EncodeSenseData ( self, ballX, ballY, paddleAY, paddleBY, centerX, centerY, timeStep ):
    # Encodes sense data as an SDR and returns it.

        # Encode bit representations for center of attention view.
        centerBitsX  = self.centerEncoderX.encode( centerX )
        centerBitsY  = self.centerEncoderY.encode( centerY )

        # Encode bit representations for ball bits.
        if ballX != None and ballY != None and Within( ballX - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ) and Within( ballY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), True ):
            ballBitsX    = self.ballEncoderX.encode( ballX - centerX )
            ballBitsY    = self.ballEncoderY.encode( ballY - centerY  )
            ballBitsT    = self.ballEncoderT.encode( timeStep )
        else:
            ballBitsX    = SDR( self.ballEncoderX.size )
            ballBitsY    = SDR( self.ballEncoderY.size )
            ballBitsT    = SDR( self.ballEncoderT.size )

        # Encode bit representations for paddle A bits.
        if paddleAY != None and Within( -350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ) and Within( paddleAY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), True ):
            paddleABitsY = self.paddleEncoderY.encode( paddleAY - centerY )
        else:
            paddleABitsY = SDR( self.paddleEncoderY.size )

        # Encode bit representations for paddle B bits.
        if paddleBY != None and Within( 350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ) and Within( paddleBY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), True ):
            paddleBBitsY = self.paddleEncoderY.encode( paddleBY - centerY )
        else:
            paddleBBitsY = SDR( self.paddleEncoderY.size )

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( self.encodingWidth ).concatenate( [ ballBitsX, ballBitsY, ballBitsT, centerBitsX, centerBitsY, paddleABitsY, paddleBBitsY ] )
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    def EncodeItLearnIt( self, ballX, ballY, paddleAY, paddleBY, centerX, centerY, chosenMotor, timeStep, learnIt ):
    # Performs learning on the 1-step of the 3-step sequence.

        thisSDR = self.EncodeSenseData( ballX, ballY, paddleAY, paddleBY, centerX, centerY, timeStep )

        # Feed SDR into tp.
        self.tp.compute( thisSDR, learn = learnIt )
        self.tp.activateDendrites( learn = learnIt )
        winnerCellsTP = self.tp.getWinnerCells()

        # Feed x and y position into classifier to learn ball and paddle positions.
        # Classifier can only take positive input, so need to transform origin.
        if learnIt:
            if ballX !=  None and ballY != None and Within( ballX - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( ballY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
                self.ballLocalXClass.learn( pattern = thisSDR, classification = int( ballX - centerX + ( self.localDimX / 2 ) ) )
                self.ballLocalYClass.learn( pattern = thisSDR, classification = int( ballY - centerY + ( self.localDimY / 2 ) ) )

                self.ballGlobalXClass.learn( pattern = winnerCellsTP, classification = int( ballX + ( self.screenWidth / 2 ) ) )
                self.ballGlobalYClass.learn( pattern = winnerCellsTP, classification = int( ballY + ( self.screenHeight / 2 ) ) )
                self.ballGlobalTClass.learn( pattern = winnerCellsTP, classification = timeStep )

            if paddleAY != None and Within( -350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( paddleAY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
                self.paddleALocalYClass.learn( pattern = thisSDR, classification = int( paddleAY - centerY + ( self.localDimY / 2 ) ) )
                self.paddleAMotorClass.learn( pattern = thisSDR, classification = chosenMotor )
            if paddleBY != None and Within( 350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( paddleBY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
                self.paddleBLocalYClass.learn( pattern = thisSDR, classification = int( paddleBY - centerY + ( self.localDimY / 2 ) ) )
                self.paddleBMotorClass.learn( pattern = thisSDR, classification = chosenMotor )

    def LearnTimeStepBall ( self, secondLast, last, present, jump ):
    # Learn the three time-step data for ball, centered around assigned center position.

        self.tp.reset()
        self.EncodeItLearnIt( secondLast[ 0 ], secondLast[ 1 ], None, None, last[ 0 ], last[ 1 ], None, 0, True )
        self.EncodeItLearnIt( last[ 0 ], last[ 1 ], None, None, last[ 0 ], last[ 1 ], None, 1, True )
        self.EncodeItLearnIt( present[ 0 ], present[ 1 ], None, None, last[ 0 ], last[ 1 ], None, 2, True )
        for step in range( 3, self.maxMemoryDist ):
            self.EncodeItLearnIt( self.memBuffer[ step ][ 0 ], self.memBuffer[ step ][ 1 ], None, None, self.memBuffer[ step ][ 0 ], self.memBuffer[ step ][ 1 ], None, step, True )

    def LearnTimeStepPaddle ( self, last, present, centerX, centerY, chosenMotor ):
    # Learn motion of paddle in tune with chosen motor function.

        self.tp.reset()
        self.EncodeItLearnIt( None, None, last[ 2 ], last[ 3 ], centerX, centerY, chosenMotor, 0, True )
        self.EncodeItLearnIt( None, None, present[ 2 ], present[ 3 ], centerX, centerY, chosenMotor, 1, True )

    def PredictTimeStepBall ( self, secondLast, last, present ):
    # Train time-step data, from secondlast to last, centered around last, and then predict next position and return.

        self.tp.reset()
        self.EncodeItLearnIt( secondLast[ 0 ], secondLast[ 1 ], None, None, last[ 0 ], last[ 1 ], None, 0, False )
        self.EncodeItLearnIt( last[ 0 ], last[ 1 ], None, None, last[ 0 ], last[ 1 ], None, 1, False )
        self.EncodeItLearnIt( present[ 0 ], present[ 1 ], None, None, last[ 0 ], last[ 1 ], None, 2, False )

        for step in range( 0, self.maxMemoryDist - 3 ):
            predictCellsTP = self.tp.getPredictiveCells()

            # Get predicted location for next time step.
            positionX = numpy.argmax( self.ballGlobalXClass.infer( pattern = predictCellsTP ) ) - int( self.screenWidth / 2 )
            positionY = numpy.argmax( self.ballGlobalYClass.infer( pattern = predictCellsTP ) ) - int( self.screenHeight / 2 )

            self.predPositions.append( [ positionX, positionY, None, None ] )

            stepSenseSDR = SDR( self.sp.getColumnDimensions() )
            stepSenseSDR.sparse = numpy.unique( [ self.tp.columnForCell( cell ) for cell in predictCellsTP.sparse ] )
            self.tp.compute( stepSenseSDR, learn = False )
            self.tp.activateDendrites( learn = False )

    def Brain ( self, ballX, ballY, paddleAY, paddleBY ):
    # Agents brain center.

        # Set present ball coordinates for next time-step.
        self.memBuffer.append( [ ballX, ballY, paddleAY, paddleBY, 1, 1 ] )
        while len( self.memBuffer ) > self.maxMemoryDist:
            self.memBuffer.pop( 0 )

        # Learn 3-step sequence, and jump, centered around ball. Do this back in memory.
        self.LearnTimeStepBall( self.memBuffer[ 0 ], self.memBuffer[ 1 ], self.memBuffer[ 2 ], self.memBuffer[ -1 ] )

        # Predict jump position.
        self.PredictTimeStepBall( self.memBuffer[ -3 ], self.memBuffer[ -2 ], self.memBuffer[ -1 ] )
        while len( self.predPositions ) > self.maxPredLocations:
            self.predPositions.pop( 0 )

        paddleADirect = False
        paddleBDirect = False
        chosenMotorA = 1
        chosenMotorB = 1

        # If ball is predicted to fall off then try to move paddle there.
        for nextPosition in self.predPositions:
            if nextPosition[ 0 ] <= -330:
                paddleADirect = True
                if paddleAY > nextPosition[ 1 ]:
                    chosenMotorA = 2
                    self.memBuffer[ -1 ][ 4 ] = chosenMotorA
                elif paddleAY < nextPosition[ 1 ]:
                    chosenMotorA = 0
                    self.memBuffer[ -1 ][ 4 ] = chosenMotorA
                break
            elif nextPosition[ 0 ] >= 330:
                paddleBDirect = True
                if paddleBY > nextPosition[ 1 ]:
                    chosenMotorB = 2
                    self.memBuffer[ -1 ][ 5 ] = chosenMotorB
                elif paddleBY < nextPosition[ 1 ]:
                    chosenMotorB = 0
                    self.memBuffer[ -1 ][ 5 ] = chosenMotorB
                break

        # If we aren't directing paddles movement then just learn paddle movement with random motion.
        if not paddleADirect:
            chosenMotorA = random.choice( [ 0, 1, 2 ] )
            self.memBuffer[ -1 ][ 4 ] = chosenMotorA
        if not paddleBDirect:
            chosenMotorB = random.choice( [ 0, 1, 2 ] )
            self.memBuffer[ -1 ][ 5 ] = chosenMotorB

        # Learn last 2-step sequence centered around paddle_a.
        self.LearnTimeStepPaddle( self.memBuffer[ -2 ], self.memBuffer[ -1 ], -350, self.memBuffer[ -2 ][ 2 ], self.memBuffer[ -2 ][ 4 ] )
        # Learn last 2-step sequence centered around paddle_b.
        self.LearnTimeStepPaddle( self.memBuffer[ -2 ], self.memBuffer[ -1 ], 350, self.memBuffer[ -2 ][ 3 ], self.memBuffer[ -2 ][ 5 ] )

#        return [ chosenMotorA, chosenMotorB ]
        return [ 1, 1 ]
