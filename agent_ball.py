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

    maxPredLocations = 7
    maxMemoryDist = 10

    def __init__( self, name, screenHeight, screenWidth, ballHeight, ballWidth, paddleHeight, paddleWidth ):

        self.ID = name

        self.screenHeight = screenHeight
        self.screenWidth  = screenWidth

        # Set up encoder parameters
        ballXEncodeParams    = ScalarEncoderParameters()
        ballYEncodeParams    = ScalarEncoderParameters()
        centerXEncodeParams   = ScalarEncoderParameters()
        centerYEncodeParams   = ScalarEncoderParameters()
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
        self.centerEncoderX  = ScalarEncoder( centerXEncodeParams )
        self.centerEncoderY  = ScalarEncoder( centerYEncodeParams )
        self.paddleEncoderY  = ScalarEncoder( paddleEncodeParams )

        self.encodingWidth = ( self.ballEncoderX.size + self.ballEncoderY.size + self.centerEncoderX.size +
            self.centerEncoderY.size + ( self.paddleEncoderY.size * 2 ) )

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
        self.ballLocalTClass    = Classifier( alpha = 1 )

        self.paddleALocalYClass = Classifier( alpha = 1 )
        self.paddleBLocalYClass = Classifier( alpha = 1 )

        self.paddleAMotorClass  = Classifier( alpha = 1 )
        self.paddleBMotorClass  = Classifier( alpha = 1 )

        self.predPositions = []
        self.memBuffer = [ [ 0, 0, 0, 0, 0, 0, 1, 1 ] ] * self.maxMemoryDist

    def EncodeSenseData ( self, ballX, ballY, paddleAY, paddleBY, centerX, centerY ):
    # Encodes sense data as an SDR and returns it.

        # Encode bit representations for center of attention view.
        centerBitsX  = self.centerEncoderX.encode( centerX )
        centerBitsY  = self.centerEncoderY.encode( centerY )

        # Encode bit representations for ball bits.
        if ballX != None and ballY != None and Within( ballX - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ) and Within( ballY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), True ):
            ballBitsX    = self.ballEncoderX.encode( ballX - centerX )
            ballBitsY    = self.ballEncoderY.encode( ballY - centerY  )
        else:
            ballBitsX    = SDR( self.ballEncoderX.size )
            ballBitsY    = SDR( self.ballEncoderY.size )

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
        encoding = SDR( self.encodingWidth ).concatenate( [ ballBitsX, ballBitsY, centerBitsX, centerBitsY, paddleABitsY, paddleBBitsY ] )
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    def EncodeItLearnIt( self, ballX, ballY, ballVelX, ballVelY, paddleAY, paddleBY, centerX, centerY, chosenMotor, learnIt ):
    # Performs learning on the 1-step of the 3-step sequence.

        thisSDR = self.EncodeSenseData( ballX, ballY, paddleAY, paddleBY, centerX, centerY )

        # Feed SDR into tp.
        self.tp.compute( thisSDR, learn = learnIt )
        self.tp.activateDendrites( learn = learnIt )
        activeCellsTP = self.tp.getActiveCells()

        # Feed x and y position into classifier to learn ball and paddle positions.
        # Classifier can only take positive input, so need to transform origin.
        if learnIt:
            if ballX !=  None and ballY != None and Within( ballX - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( ballY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
                self.ballLocalXClass.learn( pattern = activeCellsTP, classification = ballVelX + 100 )
                self.ballLocalYClass.learn( pattern = activeCellsTP, classification = ballVelY + 100 )

            if paddleAY != None and Within( -350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( paddleAY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
                self.paddleALocalYClass.learn( pattern = thisSDR, classification = int( paddleAY - centerY + ( self.localDimY / 2 ) ) )
                self.paddleAMotorClass.learn( pattern = thisSDR, classification = chosenMotor )
            if paddleBY != None and Within( 350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( paddleBY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
                self.paddleBLocalYClass.learn( pattern = thisSDR, classification = int( paddleBY - centerY + ( self.localDimY / 2 ) ) )
                self.paddleBMotorClass.learn( pattern = thisSDR, classification = chosenMotor )

    def EncodeItPredictIt( self ):
    # Run the predictive cells through tp to get the next predictive cells, and extract the position displacement.

        predictCellsTP = self.tp.getPredictiveCells()



    def LearnTimeStepBall ( self ):
    # Learn the three time-step data for ball, centered around assigned center position.

        self.tp.reset()
        for step in range( int( self.maxMemoryDist / 3 ) ):
            self.EncodeItLearnIt( self.memBuffer[ ( step * 3 ) ][ 0 ], self.memBuffer[ ( step * 3 ) ][ 1 ], self.memBuffer[ ( step * 3 ) ][ 2 ], self.memBuffer[ ( step * 3 ) ][ 3 ], None, None, self.memBuffer[ ( step * 3 ) + 1 ][ 0 ], self.memBuffer[ ( step * 3 ) + 1 ][ 1 ], None, True )
            self.EncodeItLearnIt( self.memBuffer[ ( step * 3 ) + 1 ][ 0 ], self.memBuffer[ ( step * 3 ) + 1 ][ 1 ], self.memBuffer[ ( step * 3 ) + 1 ][ 2 ], self.memBuffer[ ( step * 3 ) + 1 ][ 3 ], None, None, self.memBuffer[ ( step * 3 ) + 1 ][ 0 ], self.memBuffer[ ( step * 3 ) + 1 ][ 1 ], None, True )
            self.EncodeItLearnIt( self.memBuffer[ ( step * 3 ) + 2 ][ 0 ], self.memBuffer[ ( step * 3 ) + 2 ][ 1 ], self.memBuffer[ ( step * 3 ) + 2 ][ 2 ], self.memBuffer[ ( step * 3 ) + 2 ][ 3 ], None, None, self.memBuffer[ ( step * 3 ) + 1 ][ 0 ], self.memBuffer[ ( step * 3 ) + 1 ][ 1 ], None, True )

    def PredictTimeStepBall ( self ):
    # Feed in initial sequence and then run through to all position predictions.

        self.tp.reset()
        # Feed in initial sequence, most recent three steps.
        self.EncodeItLearnIt( self.memBuffer[ -3 ][ 0 ], self.memBuffer[ -3 ][ 1 ], self.memBuffer[ -3 ][ 2 ], self.memBuffer[ -3 ][ 3 ], None, None, self.memBuffer[ -2 ][ 0 ], self.memBuffer[ -2 ][ 1 ], None, False )
        self.EncodeItLearnIt( self.memBuffer[ -2 ][ 0 ], self.memBuffer[ -2 ][ 1 ], self.memBuffer[ -2 ][ 2 ], self.memBuffer[ -2 ][ 3 ], None, None, self.memBuffer[ -2 ][ 0 ], self.memBuffer[ -2 ][ 1 ], None, False )
        self.EncodeItLearnIt( self.memBuffer[ -1 ][ 0 ], self.memBuffer[ -1 ][ 1 ], self.memBuffer[ -1 ][ 2 ], self.memBuffer[ -1 ][ 3 ], None, None, self.memBuffer[ -2 ][ 0 ], self.memBuffer[ -2 ][ 1 ], None, False )

        # Predict the remaining steps in the sequence
        for step in range( self.maxPredLocations ):
            nextPosition = self.PredictTimeStep( self.predPositions[ -2 ], self.predPositions[ -1 ] )
            if Within( nextPosition[ 0 ], -self.screenWidth / 2, self.screenWidth / 2, True ):
                if Within( nextPosition[ 1 ], -self.screenHeight / 2, self.screenHeight / 2, True ):
                    self.predPositions.append( nextPosition )
                else:
                    break
            else:
                break

        secondLastSDR = self.EncodeSenseData( secondLast[ 0 ], secondLast[ 1 ], secondLast[ 2 ], secondLast[ 3 ], last[ 0 ], last[ 1 ] )
        self.tp.compute( secondLastSDR, learn = False )
        self.tp.activateDendrites( learn = False )

        lastSDR = self.EncodeSenseData( last[ 0 ], last[ 1 ], last[ 2 ], last[ 3 ], last[ 0 ], last[ 1 ] )
        self.tp.compute( lastSDR, learn = False )
        self.tp.activateDendrites( learn = False )
        predictCellsTP = self.tp.getPredictiveCells()

        # Get predicted location for next time step.
        stepSenseSDR = SDR( self.sp.getColumnDimensions() )
        stepSenseSDR.sparse = numpy.unique( [ self.tp.columnForCell( cell ) for cell in predictCellsTP.sparse ] )
        positionX = numpy.argmax( self.ballLocalXClass.infer( pattern = stepSenseSDR ) ) - int( self.localDimX / 2 )
        positionY = numpy.argmax( self.ballLocalYClass.infer( pattern = stepSenseSDR ) ) - int( self.localDimY / 2 )

        return [ last[ 0 ] + positionX, last[ 1 ] + positionY, None, None ]

    def LearnTimeStepPaddle ( self, last, present, centerX, centerY, chosenMotor ):
    # Learn motion of paddle in tune with chosen motor function.

        self.tp.reset()
        self.EncodeItLearnIt( None, None, last[ 4 ], last[ 5 ], centerX, centerY, chosenMotor )
        self.EncodeItLearnIt( None, None, present[ 4 ], present[ 5 ], centerX, centerY, chosenMotor )

    def Brain ( self, ballX, ballY, ballVelX, ballVelY, paddleAY, paddleBY ):
    # Agents brain center.

        # Set present ball coordinates for next time-step.
        self.memBuffer.append( [ ballX, ballY, ballVelX, ballVelY, paddleAY, paddleBY, 1, 1 ] )
        while len( self.memBuffer ) > self.maxMemoryDist:
            self.memBuffer.pop( 0 )

        # Learn 3-step sequence, and jump, centered around ball. Do this back in memory.
        self.LearnTimeStepBall()

        # Store last and present locations.
        self.predPositions.clear()
        self.predPositions.append( self.memBuffer[ -2 ] )
        self.predPositions.append( self.memBuffer[ -1 ] )

        # Predict next time step locations and store them in predPositions.
        self.PredictTimeStepBall()

        paddleADirect = False
        paddleBDirect = False
        chosenMotorA = 1
        chosenMotorB = 1

        # If ball is predicted to fall off then try to move paddle there.
        if self.predPositions[ -1 ][ 0 ] <= -330:
            paddleADirect = True
            if paddleAY > self.predPositions[ -1 ][ 1 ]:
                chosenMotorA = 2
                self.memBuffer[ -1 ][ 6 ] = chosenMotorA
            elif paddleAY < self.predPositions[ -1 ][ 1 ]:
                chosenMotorA = 0
                self.memBuffer[ -1 ][ 6 ] = chosenMotorA
            break
        elif self.predPositions[ -1 ][ 0 ] >= 330:
            paddleBDirect = True
            if paddleBY > self.predPositions[ -1 ][ 1 ]:
                chosenMotorB = 2
                self.memBuffer[ -1 ][ 7 ] = chosenMotorB
            elif paddleBY < self.predPositions[ -1 ][ 1 ]:
                chosenMotorB = 0
                self.memBuffer[ -1 ][ 7 ] = chosenMotorB
            break

        # If we aren't directing paddles movement then just learn paddle movement with random motion.
        if not paddleADirect:
            chosenMotorA = random.choice( [ 0, 1, 2 ] )
            self.memBuffer[ -1 ][ 6 ] = chosenMotorA
        if not paddleBDirect:
            chosenMotorB = random.choice( [ 0, 1, 2 ] )
            self.memBuffer[ -1 ][ 7 ] = chosenMotorB

        # Learn last 2-step sequence centered around paddle_a.
        self.LearnTimeStepPaddle( self.memBuffer[ -2 ], self.memBuffer[ -1 ], -350, self.memBuffer[ -2 ][ 4 ], self.memBuffer[ -2 ][ 6 ] )
        # Learn last 2-step sequence centered around paddle_b.
        self.LearnTimeStepPaddle( self.memBuffer[ -2 ], self.memBuffer[ -1 ], 350, self.memBuffer[ -2 ][ 5 ], self.memBuffer[ -2 ][ 7 ] )

        return [ chosenMotorA, chosenMotorB ]
