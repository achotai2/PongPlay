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

    maxPredLocations = 20
    maxMemoryDist = 20

    def __init__( self, name, screenHeight, screenWidth, ballHeight, ballWidth, paddleHeight, paddleWidth ):

        self.ID = name

        self.screenHeight = screenHeight
        self.screenWidth  = screenWidth

        # Set up encoder parameters
        ballXEncodeParams    = ScalarEncoderParameters()
        ballYEncodeParams    = ScalarEncoderParameters()
        ballTEncodeParams    = ScalarEncoderParameters()
        paddleEncodeParams   = ScalarEncoderParameters()
        wallXEncodeParams    = ScalarEncoderParameters()
        wallYEncodeParams    = ScalarEncoderParameters()
        ballVelEncodeParams  = ScalarEncoderParameters()

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

        ballVelEncodeParams.activeBits = 11
        ballVelEncodeParams.radius     = 5
        ballVelEncodeParams.clipInput  = False
        ballVelEncodeParams.minimum    = -50
        ballVelEncodeParams.maximum    = 50
        ballVelEncodeParams.periodic   = False

        # Set up encoders
        self.ballEncoderX    = ScalarEncoder( ballXEncodeParams )
        self.ballEncoderY    = ScalarEncoder( ballYEncodeParams )
        self.paddleEncoderY  = ScalarEncoder( paddleEncodeParams )
        self.wallEncoderX    = ScalarEncoder( wallXEncodeParams )
        self.wallEncoderY    = ScalarEncoder( wallYEncodeParams )
        self.ballVelEncoder  = ScalarEncoder( ballVelEncodeParams )

        self.encodingWidth = ( self.ballEncoderX.size + self.ballEncoderY.size + ( self.ballVelEncoder.size * 2 ) +
            ( self.paddleEncoderY.size * 2 ) + self.wallEncoderX.size + self.wallEncoderY.size )

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

        self.ballShiftXClass    = Classifier( alpha = 1 )
        self.ballShiftYClass    = Classifier( alpha = 1 )
        self.ballVelXClass      = Classifier( alpha = 1 )
        self.ballVelYClass      = Classifier( alpha = 1 )

        self.paddleALocalYClass = Classifier( alpha = 1 )
        self.paddleBLocalYClass = Classifier( alpha = 1 )
        self.paddleAMotorClass  = Classifier( alpha = 1 )
        self.paddleBMotorClass  = Classifier( alpha = 1 )

        self.predPositions = []
        self.memBuffer = [ [ 0, 0, 0, 0, 1, 1, 0, 0 ] ] * self.maxMemoryDist

    def EncodeSenseData ( self, ballX, ballY, ballVelX, ballVelY, paddleAY, paddleBY, centerX, centerY ):
    # Encodes sense data as an SDR and returns it.

        # Encode bit representations for ball bits.
        if ballVelX != None and ballVelY != None and Within( ballVelX, -50, 50, True ) and Within( ballVelY, -50, 50, True ):
            ballVelBitsX    = self.ballVelEncoder.encode( ballVelX )
            ballVelBitsY    = self.ballVelEncoder.encode( ballVelY )
        else:
            ballVelBitsX    = SDR( self.ballVelEncoder.size )
            ballVelBitsY    = SDR( self.ballVelEncoder.size )

        # Encode bit representations for ball bits.
        if ballX != None and ballY != None and Within( ballX - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ) and Within( ballY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), True ):
            ballBitsX    = self.ballEncoderX.encode( ballX - centerX )
            ballBitsY    = self.ballEncoderY.encode( ballY - centerY )
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

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( self.encodingWidth ).concatenate( [ ballBitsX, ballBitsY, paddleABitsY, paddleBBitsY, wallBitsX, wallBitsY, ballVelBitsX, ballVelBitsY ] )
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    def EncodeItLearnIt( self, ballIdx, nextBallIdx, paddleIdx, centerX, centerY, chosenMotor, learnIt ):
    # Performs learning on the 1-step of the 3-step sequence.

        if ballIdx != None:
            ballX = self.memBuffer[ ballIdx ][ 0 ]
            ballY = self.memBuffer[ ballIdx ][ 1 ]
            ballDispX = self.memBuffer[ nextBallIdx ][ 0 ] - ballX
            ballDispY = self.memBuffer[ nextBallIdx ][ 1 ] - ballY
            ballVelX = self.memBuffer[ ballIdx ][ 6 ]
            ballVelY = self.memBuffer[ ballIdx ][ 7 ]
        else:
            ballX = None
            ballY = None
            ballDispX = None
            ballDispY = None
            ballVelX = None
            ballVelY = None

        if paddleIdx != None:
            paddleAY = self.memBuffer[ paddleIdx ][ 2 ]
            paddleBY = self.memBuffer[ paddleIdx ][ 3 ]
        else:
            paddleAY = None
            paddleBY = None

        thisSDR = self.EncodeSenseData( ballX, ballY, ballVelX, ballVelY, paddleAY, paddleBY, centerX, centerY )

        # Feed SDR into tp.
        self.tp.compute( thisSDR, learn = learnIt )
        self.tp.activateDendrites( learn = learnIt )
        winnerCellsTP = self.tp.getWinnerCells()

        # Feed x and y position into classifier to learn ball and paddle positions.
        # Classifier can only take positive input, so need to transform origin.
        if learnIt:
            if ballX !=  None and ballY != None:
                self.ballShiftXClass.learn( pattern = winnerCellsTP, classification = ballDispX + 100 )
                self.ballShiftYClass.learn( pattern = winnerCellsTP, classification = ballDispY + 100 )
                self.ballVelXClass.learn( pattern = winnerCellsTP, classification = ballVelX + 100 )
                self.ballVelYClass.learn( pattern = winnerCellsTP, classification = ballVelY + 100 )

            if paddleAY != None and Within( -350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( paddleAY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
                self.paddleALocalYClass.learn( pattern = thisSDR, classification = int( paddleAY - centerY + ( self.localDimY / 2 ) ) )
                self.paddleAMotorClass.learn( pattern = thisSDR, classification = chosenMotor )
            if paddleBY != None and Within( 350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( paddleBY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
                self.paddleBLocalYClass.learn( pattern = thisSDR, classification = int( paddleBY - centerY + ( self.localDimY / 2 ) ) )
                self.paddleBMotorClass.learn( pattern = thisSDR, classification = chosenMotor )

    def LearnTimeStepBall ( self ):
    # Learn the three time-step data for ball, centered around assigned center position.

        self.tp.reset()
        centerX = self.memBuffer[ 1 ][ 0 ]
        centerY = self.memBuffer[ 1 ][ 1 ]
        self.EncodeItLearnIt( 0, 1, None, centerX, centerY, None, True )
        self.EncodeItLearnIt( 1, 2, None, centerX, centerY, None, True )
        self.EncodeItLearnIt( 2, 3, None, centerX, centerY, None, True )

        for step in range( 3, self.maxMemoryDist - 1 ):
            centerX = self.memBuffer[ step ][ 0 ]
            centerY = self.memBuffer[ step ][ 1 ]
            self.EncodeItLearnIt( step, step + 1, None, centerX, centerY, None, True )

    def LearnTimeStepPaddle ( self, last, present ):
    # Learn motion of paddle in tune with chosen motor function.

        # Paddle A.
        self.tp.reset()
        self.EncodeItLearnIt( None, None, last, -350, self.memBuffer[ last ][ 2 ], self.memBuffer[ last ][ 4 ], True )
        self.EncodeItLearnIt( None, None, present, -350, self.memBuffer[ last ][ 2 ], self.memBuffer[ last ][ 4 ], True )

        # Paddle B.
        self.tp.reset()
        self.EncodeItLearnIt( None, None, last, 350, self.memBuffer[ last ][ 3 ], self.memBuffer[ last ][ 5 ], True )
        self.EncodeItLearnIt( None, None, present, 350, self.memBuffer[ last ][ 3 ], self.memBuffer[ last ][ 5 ], True )

    def PredictTimeStepBall ( self ):
    # Train time-step data, from secondlast to last, centered around last, and then predict next position and return.

        self.tp.reset()
        # Feed in initial sequence, most recent three steps for ball.
        centerX = self.memBuffer[ -2 ][ 0 ]
        centerY = self.memBuffer[ -2 ][ 1 ]
        self.EncodeItLearnIt( -3, -2, None, centerX, centerY, None, False )
        self.EncodeItLearnIt( -2, -1, None, centerX, centerY, None, False )
        self.EncodeItLearnIt( -1, 0, None, centerX, centerY, None, False )

        # Store last and present locations.
        self.predPositions.clear()
        self.predPositions.append( [ self.memBuffer[ -2 ][ 0 ], self.memBuffer[ -2 ][ 1 ] ] )
        self.predPositions.append( [ self.memBuffer[ -1 ][ 0 ], self.memBuffer[ -1 ][ 1 ] ] )

        # Predict the remaining steps in the sequence
        for step in range( self.maxPredLocations - 3 ):
            # Get the predicted cells from tp.
            winnerCellsTP = self.tp.getWinnerCells()
            predictCellsTP = self.tp.getPredictiveCells()

            shiftX = numpy.argmax( self.ballShiftXClass.infer( pattern = winnerCellsTP ) ) - 100
            shiftY = numpy.argmax( self.ballShiftYClass.infer( pattern = winnerCellsTP ) ) - 100

            velX = numpy.argmax( self.ballVelXClass.infer( pattern = winnerCellsTP ) ) - 100
            velY = numpy.argmax( self.ballVelYClass.infer( pattern = winnerCellsTP ) ) - 100

            newBallPosX = self.predPositions[ -1 ][ 0 ] + shiftX
            if newBallPosX > self.screenWidth / 2:
                newBallPosX = int( self.screenWidth / 2 )
            elif newBallPosX < -self.screenWidth / 2:
                newBallPosX = -int( self.screenWidth / 2 )

            newBallPosY = self.predPositions[ -1 ][ 1 ] + shiftY
            if newBallPosY > self.screenHeight / 2:
                newBallPosY = int( self.screenHeight / 2 )
            elif newBallPosY < -self.screenHeight / 2:
                newBallPosY = -int( self.screenHeight / 2 )

            self.predPositions.append( [ newBallPosX, newBallPosY ] )

            stepSenseSDR = self.EncodeSenseData( newBallPosX, newBallPosY, velX, velY, None, None, newBallPosX, newBallPosY )
            self.tp.compute( stepSenseSDR, learn = False )
            self.tp.activateDendrites( learn = False )

    def Brain ( self, ballX, ballY, ballVelX, ballVelY, paddleAY, paddleBY ):
    # Agents brain center.

        # Set present ball coordinates for next time-step.
        self.memBuffer.append( [ ballX, ballY, paddleAY, paddleBY, 1, 1, ballVelX, ballVelY ] )
        while len( self.memBuffer ) > self.maxMemoryDist:
            self.memBuffer.pop( 0 )

        # Learn 3-step sequence, and jump, centered around ball. Do this back in memory.
        self.LearnTimeStepBall()

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
                self.memBuffer[ -1 ][ 4 ] = chosenMotorA
            elif paddleAY < self.predPositions[ -1 ][ 1 ]:
                chosenMotorA = 0
                self.memBuffer[ -1 ][ 4 ] = chosenMotorA
        elif self.predPositions[ -1 ][ 0 ] >= 330:
            paddleBDirect = True
            if paddleBY > self.predPositions[ -1 ][ 1 ]:
                chosenMotorB = 2
                self.memBuffer[ -1 ][ 5 ] = chosenMotorB
            elif paddleBY < self.predPositions[ -1 ][ 1 ]:
                chosenMotorB = 0
                self.memBuffer[ -1 ][ 5 ] = chosenMotorB

        # If we aren't directing paddles movement then just learn paddle movement with random motion.
        if not paddleADirect:
            chosenMotorA = random.choice( [ 0, 1, 2 ] )
            self.memBuffer[ -1 ][ 4 ] = chosenMotorA
        if not paddleBDirect:
            chosenMotorB = random.choice( [ 0, 1, 2 ] )
            self.memBuffer[ -1 ][ 5 ] = chosenMotorB

        # Learn last 2-step sequence centered around paddles.
        self.LearnTimeStepPaddle( -2, -1 )

        return [ 1, 1 ]
