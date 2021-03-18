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

    localDimX   = 150
    localDimY   = 150

    maxPredLocations = 10
    maxMemoryDist = 10

    def __init__( self, name, screenHeight, screenWidth, ballHeight, ballWidth, paddleHeight, paddleWidth ):

        self.ID = name

        self.screenHeight = screenHeight
        self.screenWidth  = screenWidth

        # Set up encoder parameters
        ballXEncodeParams    = ScalarEncoderParameters()
        ballYEncodeParams    = ScalarEncoderParameters()
        wallXEncodeParams    = ScalarEncoderParameters()
        wallYEncodeParams    = ScalarEncoderParameters()
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

        wallXEncodeParams.activeBits = 11
        wallXEncodeParams.radius     = 5
        wallXEncodeParams.clipInput  = False
        wallXEncodeParams.minimum    = -int( self.localDimX / 2 )
        wallXEncodeParams.maximum    = int( self.localDimX / 2 )
        wallXEncodeParams.periodic   = False

        wallYEncodeParams.activeBits = 11
        wallYEncodeParams.radius     = 5
        wallYEncodeParams.clipInput  = False
        wallYEncodeParams.minimum    = -int( self.localDimY / 2 )
        wallYEncodeParams.maximum    = int( self.localDimY / 2 )
        wallYEncodeParams.periodic   = False

        # Set up encoders
        self.ballEncoderX    = ScalarEncoder( ballXEncodeParams )
        self.ballEncoderY    = ScalarEncoder( ballYEncodeParams )
        self.wallEncoderX    = ScalarEncoder( wallXEncodeParams )
        self.wallEncoderY    = ScalarEncoder( wallYEncodeParams )
        self.paddleEncoderY  = ScalarEncoder( paddleEncodeParams )

        self.encodingWidth = ( self.ballEncoderX.size + self.ballEncoderY.size + self.wallEncoderX.size +
            self.wallEncoderY.size + ( self.paddleEncoderY.size * 2 ) )

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

        self.predPositions = [ [ 0, 0 ] ] * self.maxPredLocations
        self.memBuffer = [ [ 0, 0, 0, 0, 1, 1 ] ] * self.maxMemoryDist

    def EncodeSenseData ( self, ballX, ballY, paddleAY, paddleBY, centerX, centerY ):
    # Encodes sense data as an SDR and returns it.

        # Encode bit representations for wall bits.
        if Within( int( self.screenWidth / 2 ) - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ):
            wallBitsX = self.wallEncoderX.encode( int( self.screenWidth / 2 ) - centerX )
        elif Within( -int( self.screenWidth / 2 ) - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ):
            wallBitsX = self.wallEncoderX.encode( -int( self.screenWidth / 2 ) - centerX )
        else:
            wallBitsX = SDR ( self.wallEncoderX.size )
        if Within( int( self.screenHeight / 2 ) - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), True ):
            wallBitsY = self.wallEncoderY.encode( int( self.screenHeight / 2 ) - centerY )
        elif Within( -int( self.screenHeight / 2 ) - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), True ):
            wallBitsY = self.wallEncoderY.encode( -int( self.screenHeight / 2 ) - centerY )
        else:
            wallBitsY = SDR ( self.wallEncoderY.size )

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
        encoding = SDR( self.encodingWidth ).concatenate( [ ballBitsX, ballBitsY, wallBitsX, wallBitsY, paddleABitsY, paddleBBitsY ] )
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    def EncodeItLearnIt( self, ballX, ballY, paddleAY, paddleBY, centerX, centerY, chosenMotor ):
    # Performs learning on the 1-step of the 3-step sequence.

        thisSDR = self.EncodeSenseData( ballX, ballY, paddleAY, paddleBY, centerX, centerY )

        # Feed x and y position into classifier to learn ball and paddle positions.
        # Classifier can only take positive input, so need to transform origin.
        if ballX !=  None and ballY != None and Within( ballX - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( ballY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
            self.ballLocalXClass.learn( pattern = thisSDR, classification = int( ballX - centerX + ( self.localDimX / 2 ) ) )
            self.ballLocalYClass.learn( pattern = thisSDR, classification = int( ballY - centerY + ( self.localDimY / 2 ) ) )
        if paddleAY != None and Within( -350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( paddleAY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
            self.paddleALocalYClass.learn( pattern = thisSDR, classification = int( paddleAY - centerY + ( self.localDimY / 2 ) ) )
            self.paddleAMotorClass.learn( pattern = thisSDR, classification = chosenMotor )
        if paddleBY != None and Within( 350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( paddleBY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
            self.paddleBLocalYClass.learn( pattern = thisSDR, classification = int( paddleBY - centerY + ( self.localDimY / 2 ) ) )
            self.paddleBMotorClass.learn( pattern = thisSDR, classification = chosenMotor )

        # Feed SDR into tp.
        self.tp.compute( thisSDR, learn = True )
        self.tp.activateDendrites( learn = True )

    def LearnTimeStepBall ( self, secondLast, last, present ):
    # Learn the three time-step data for ball, centered around assigned center position.

        self.tp.reset()
        self.EncodeItLearnIt( secondLast[ 0 ], secondLast[ 1 ], None, None, last[ 0 ], last[ 1 ], None )
        self.EncodeItLearnIt( last[ 0 ], last[ 1 ], None, None, last[ 0 ], last[ 1 ], None )
        self.EncodeItLearnIt( present[ 0 ], present[ 1 ], None, None, last[ 0 ], last[ 1 ], None )
#        self.EncodeItLearnIt( jump[ 0 ], jump[ 1 ], None, None, jump[ 0 ], jump[ 1 ], None )

    def LearnTimeStepPaddle ( self, last, present, centerX, centerY, chosenMotor ):
    # Learn motion of paddle in tune with chosen motor function.

        self.tp.reset()
        self.EncodeItLearnIt( None, None, last[ 2 ], last[ 3 ], centerX, centerY, chosenMotor )
        self.EncodeItLearnIt( None, None, present[ 2 ], present[ 3 ], centerX, centerY, chosenMotor )

    def PredictTimeStep ( self, secondLast, last ):
    # Train time-step data, from secondlast to last, centered around last, and then predict next position and return.

        self.tp.reset()

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

    def Brain ( self, ballX, ballY, paddleAY, paddleBY ):
    # Agents brain center.

        # Set present ball coordinates for next time-step.
        self.memBuffer.append( [ ballX, ballY, paddleAY, paddleBY, 1, 1 ] )
        while len( self.memBuffer ) > self.maxMemoryDist:
            self.memBuffer.pop( 0 )

        # Learn 3-step sequence, and jump, centered around ball. Do this back in memory.
        self.LearnTimeStepBall( self.memBuffer[ -3 ], self.memBuffer[ -2 ], self.memBuffer[ -1 ] )

        # Store last and present locations.
        self.predPositions.clear()
        self.predPositions.append( self.memBuffer[ -2 ] )
        self.predPositions.append( self.memBuffer[ -1 ] )

        # Predict jump position.


        paddleADirect = False
        paddleBDirect = False
        chosenMotorA = 1
        chosenMotorB = 1

        # Predict next 10 time step locations and store them.
        for step in range( self.maxPredLocations - 2 ):
            nextPosition = self.PredictTimeStep( self.predPositions[ -2 ], self.predPositions[ -1 ] )
            if Within( nextPosition[ 0 ], -self.screenWidth / 2, self.screenWidth / 2, True ):
                if Within( nextPosition[ 1 ], -self.screenHeight / 2, self.screenHeight / 2, True ):
                    self.predPositions.append( nextPosition )

                    # If ball is predicted to fall off then try to move paddle there.
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
                else:
                    break
            else:
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

        return [ 1, 1 ]