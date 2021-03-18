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

def Overlap ( SDR1, SDR2 ):
# Computes overlap score between two passed SDRs.

    overlap = 0

    for cell1 in SDR1.sparse:
        if cell1 in SDR2.sparse:
            overlap += 1

    return overlap

def GreatestOverlap ( testSDR, listSDR, threshold ):
# Finds SDR in listSDR with greatest overlap with testSDR and returns it, and its index in the list.
# If none are found above threshold or if list is empty it returns an empty SDR of length testSDR, with index -1.

    greatest = [ SDR( testSDR.size ), -1 ]

    if len( listSDR ) > 0:
        # The first element of listSDR should always be a union of all the other SDRs in list,
        # so a check can be performed first.
        if Overlap( testSDR, listSDR[0] ) >= threshold:
            aboveThreshold = []
            for idx, checkSDR in enumerate( listSDR ):
                if idx != 0:
                    thisOverlap = Overlap( testSDR, checkSDR )
                    if thisOverlap >= threshold:
                        aboveThreshold.append( [ thisOverlap, [checkSDR, idx] ] )
            if len( aboveThreshold ) > 0:
                greatest = sorted( aboveThreshold, key = lambda tup: tup[0], reverse = True )[ 0 ][ 1 ]

    return greatest

class BallAgent:

    localDimX   = 100
    localDimY   = 100

    maxPredLocations = 10
    maxMemoryDist = 3
    maxAttentionStore = 100

    attentionThreshold = 20

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
        ballTEncodeParams.maximum    = self.maxPredLocations + 3
        ballTEncodeParams.periodic   = False

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
        self.ballEncoderT    = ScalarEncoder( ballTEncodeParams )
        self.wallEncoderX    = ScalarEncoder( wallXEncodeParams )
        self.wallEncoderY    = ScalarEncoder( wallYEncodeParams )
        self.paddleEncoderY  = ScalarEncoder( paddleEncodeParams )

        self.encodingWidth = ( self.ballEncoderX.size + self.ballEncoderY.size + self.wallEncoderX.size +
            self.wallEncoderY.size + ( self.paddleEncoderY.size * 2 ) + self.ballEncoderT.size )

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

        self.ballLocalXClass   = Classifier( alpha = 1 )
        self.ballLocalYClass   = Classifier( alpha = 1 )
        self.ballGlobalTClass   = Classifier( alpha = 1 )

        self.paddleALocalYClass = Classifier( alpha = 1 )
        self.paddleBLocalYClass = Classifier( alpha = 1 )
        self.paddleAMotorClass  = Classifier( alpha = 1 )
        self.paddleBMotorClass  = Classifier( alpha = 1 )

        self.predPositions = [ [ 0, 0, 0, 0 ] ] * self.maxPredLocations
        self.memBuffer = [ [ 0, 0, 0, 0, 1, 1 ] ] * self.maxMemoryDist
        self.winnerCellMemory  = []
        self.attentShiftMemory = []

        self.centerX = 0
        self.centerY = 0
        self.seqStep = 0

    def EncodeSenseData ( self, ballX, ballY, paddleAY, paddleBY, centerX, centerY, timeStep, learnIt ):
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
        encoding = SDR( self.encodingWidth ).concatenate( [ ballBitsX, ballBitsY, ballBitsT, wallBitsX, wallBitsY, paddleABitsY, paddleBBitsY ] )
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, learnIt, senseSDR )

        # Feed SDR into tp.
        self.tp.compute( senseSDR, learn = learnIt )
        self.tp.activateDendrites( learn = learnIt )

        return senseSDR

    def ClassifyBall( self, classSDR, ballX, ballY, centerX, centerY, timeStep ):
    # Performs learning on the 1-step of the 3-step sequence.

        # Feed x and y position into classifier to learn ball and paddle positions.
        # Classifier can only take positive input, so need to transform origin.
        if Within( ballX - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( ballY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
            self.ballLocalXClass.learn( pattern = classSDR, classification = int( ballX - centerX + ( self.localDimX / 2 ) ) )
            self.ballLocalYClass.learn( pattern = classSDR, classification = int( ballY - centerY + ( self.localDimY / 2 ) ) )
            self.ballGlobalTClass.learn( pattern = classSDR, classification = timeStep )

    def ClassifyPaddle( self, thisSDR, paddleAY, paddleBY, centerX, centerY, chosenMotor ):

        if Within( -350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( paddleAY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
            self.paddleALocalYClass.learn( pattern = thisSDR, classification = int( paddleAY - centerY + ( self.localDimY / 2 ) ) )
            self.paddleAMotorClass.learn( pattern = thisSDR, classification = chosenMotor )
        if Within( 350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), False ) and Within( paddleBY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), False ):
            self.paddleBLocalYClass.learn( pattern = thisSDR, classification = int( paddleBY - centerY + ( self.localDimY / 2 ) ) )
            self.paddleBMotorClass.learn( pattern = thisSDR, classification = chosenMotor )

    def LearnInitialStepBall ( self ):
    # Learn the three time-step data for ball, centered around assigned center position.

        self.tp.reset()

        self.centerX = self.memBuffer[ -2 ][ 0 ]
        self.centerY = self.memBuffer[ -2 ][ 1 ]

        thisSDR = self.EncodeSenseData( self.memBuffer[ -3 ][ 0 ], self.memBuffer[ -3 ][ 1 ],  None, None, self.centerX, self.centerY, 0, True )
        self.ClassifyBall( thisSDR, self.memBuffer[ -3 ][ 0 ], self.memBuffer[ -3 ][ 1 ], self.centerX, self.centerY, 0 )

        thisSDR = self.EncodeSenseData( self.memBuffer[ -2 ][ 0 ], self.memBuffer[ -2 ][ 1 ], None, None, self.centerX, self.centerY, 1, True )
        self.ClassifyBall( thisSDR, self.memBuffer[ -2 ][ 0 ], self.memBuffer[ -2 ][ 1 ], self.centerX, self.centerY, 1 )

        thisSDR = self.EncodeSenseData( self.memBuffer[ -1 ][ 0 ], self.memBuffer[ -1 ][ 1 ], None, None, self.centerX, self.centerY, 2, True )
        self.ClassifyBall( thisSDR, self.memBuffer[ -1 ][ 0 ], self.memBuffer[ -1 ][ 1 ], self.centerX, self.centerY, 2 )

        self.seqStep = 2

    def LearnNextStepBall( self ):

        # Move center using prediction.
        self.centerX = self.predPositions[ 0 ][ 2 ]
        self.centerY = self.predPositions[ 0 ][ 3 ]
        self.seqStep += 1

        # Learn ball in last position with new center.
        thisSDR = self.EncodeSenseData( self.memBuffer[ -2 ][ 0 ], self.memBuffer[ -2 ][ 1 ],  None, None, self.centerX, self.centerY, self.seqStep, True )
        self.ClassifyBall( thisSDR, self.memBuffer[ -2 ][ 0 ], self.memBuffer[ -2 ][ 1 ], self.centerX, self.centerY, self.seqStep )

        # Learn ball in current position and new center.
        thisSDR = self.EncodeSenseData( self.predPositions[ 0 ][ 0 ], self.predPositions[ 0 ][ 1 ],  None, None, self.centerX, self.centerY, self.seqStep, True )
        self.ClassifyBall( thisSDR, self.predPositions[ 0 ][ 0 ], self.predPositions[ 0 ][ 1 ], self.centerX, self.centerY, self.seqStep )

    def LearnTimeStepPaddle ( self, last, present, centerX, centerY, chosenMotor ):
    # Learn motion of paddle in tune with chosen motor function.

        self.tp.reset()
        thisSDR = self.EncodeSenseData( None, None,last[ 2 ], last[ 3 ], centerX, centerY, 0, True )
        self.ClassifyPaddle( thisSDR, last[ 2 ], last[ 3 ], centerX, centerY, chosenMotor )
        thisSDR = self.EncodeSenseData( None, None, present[ 2 ], present[ 3 ], centerX, centerY, 1, True )
        self.ClassifyPaddle( thisSDR, present[ 2 ], present[ 3 ], centerX, centerY, chosenMotor )

    def PredictTimeStepBall ( self ):
    # Train time-step data, from secondlast to last, centered around last, and then predict next position and return.

        # Start the prediction by looking at the 3-most recent observations.
        self.tp.reset()
        centerX = self.memBuffer[ -2 ][ 0 ]
        centerY = self.memBuffer[ -2 ][ 1 ]
        self.EncodeSenseData( self.memBuffer[ -3 ][ 0 ], self.memBuffer[ -3 ][ 1 ], None, None, centerX, centerY, 0, False )
        self.EncodeSenseData( self.memBuffer[ -2 ][ 0 ], self.memBuffer[ -2 ][ 1 ], None, None, centerX, centerY, 1, False )
        self.EncodeSenseData( self.memBuffer[ -1 ][ 0 ], self.memBuffer[ -1 ][ 1 ], None, None, centerX, centerY, 2, False )

        ballX = self.memBuffer[ -1 ][ 0 ]
        ballY = self.memBuffer[ -1 ][ 1 ]

        for step in range( 3, self.maxPredLocations + 3 ):
            # Use the winner cells to find index in our motorMemory list.
            winnerCellsTP = self.tp.getWinnerCells()

            actMotor = GreatestOverlap( winnerCellsTP, self.winnerCellMemory, self.attentionThreshold )
            if len( self.winnerCellMemory ) == 0:
                self.winnerCellMemory.append( winnerCellsTP )
                self.attentShiftMemory.append( [ 50 ] * self.screenWidth * self.screenHeight )

            if actMotor[ 1 ] == -1:
                # If there isn't one then create it.
                self.winnerCellMemory.append( winnerCellsTP )
                self.winnerCellMemory[ 0 ].sparse = numpy.union1d( self.winnerCellMemory[ 0 ].sparse, winnerCellsTP.sparse )
                self.attentShiftMemory.append( [ 50 ] * self.screenWidth * self.screenHeight )

                # Keep length of winnerCellMemory bellow maxAttentionStore by removing old ones.
                while len( self.winnerCellMemory ) > self.maxAttentionStore:
                    self.winnerCellMemory.pop( 1 )
                    self.attentShiftMemory.pop( 1 )
                self.winnerCellMemory[ 0 ] = SDR( winnerCellsTP.size )
                for i in self.winnerCellMemory:
                    self.winnerCellMemory[ 0 ].sparse = numpy.union1d( self.winnerCellMemory[ 0 ].sparse, i.sparse )     # Update union element.

                actMotor = [ winnerCellsTP, len( self.winnerCellMemory ) - 1 ]
            else:
                # If there is then move it to the end of the list to keep track of recency (remove old).
                self.winnerCellMemory.append( winnerCellsTP )
                self.attentShiftMemory.append( self.attentShiftMemory[ actMotor[ 1 ] ] )
                self.winnerCellMemory.pop( actMotor[ 1 ] )
                self.attentShiftMemory.pop( actMotor[ 1 ] )

                actMotor[ 1 ] = len( self.winnerCellMemory ) - 1

            # Use shiftArray as weights to choose a random shift of attention.
            shiftIndex = random.choices( [ i for i in range( self.screenHeight * self.screenWidth ) ], weights = self.attentShiftMemory[ actMotor[ 1 ] ], k = 1 )[ 0 ]
            shiftX = int( ( shiftIndex % self.screenWidth ) - ( self.screenWidth / 2 ) )
            shiftY = int( ( ( shiftIndex - shiftX ) / self.screenWidth ) - ( self.screenHeight / 2 ) )

            # Update centerX and centerY using shifts.
            centerX += shiftX
            centerY += shiftY
            if centerX < -self.screenWidth / 2:
                centerX = int( -self.screenWidth / 2 )
            elif centerX > self.screenWidth / 2:
                centerX = int( self.screenWidth / 2 )
            if centerY < -self.screenHeight / 2:
                centerY = int( -self.screenHeight / 2 )
            elif centerY > self.screenHeight / 2:
                centerY = int( self.screenHeight / 2 )

            # Run tp using shifted center.
            self.EncodeSenseData( ballX, ballY, None, None, centerX, centerY, step, False )

            # Get predicted new local ball position.
            predictCellsTP = self.tp.getPredictiveCells()
            stepSenseSDR = SDR( self.sp.getColumnDimensions() )
            stepSenseSDR.sparse = numpy.unique( [ self.tp.columnForCell( cell ) for cell in predictCellsTP.sparse ] )
            localXInfer = self.ballLocalXClass.infer( pattern = stepSenseSDR )
            localYInfer = self.ballLocalYClass.infer( pattern = stepSenseSDR )
            if len( localXInfer ) > 0 and len( localYInfer ) > 0:
                localPosX = numpy.argmax( localXInfer ) - int( self.localDimX / 2 )
                localPosY = numpy.argmax( localYInfer ) - int( self.localDimY / 2 )
            else:
                localPosX = 0
                localPosY = 0
            ballX = localPosX + centerX
            ballY = localPosY + centerY

            # Store new ball position and center position.
            self.predPositions.append( [ ballX, ballY, centerX, centerY ] )

            # Run tp again using new ball position.
            self.EncodeSenseData( ballX, ballY, None, None, centerX, centerY, step, False )

    def Brain ( self, ballX, ballY, paddleAY, paddleBY ):
    # Agents brain center.

        # Set present ball coordinates for next time-step.
        self.memBuffer.append( [ ballX, ballY, paddleAY, paddleBY, 1, 1 ] )
        while len( self.memBuffer ) > self.maxMemoryDist:
            self.memBuffer.pop( 0 )

        # Check if the first predicted position is the balls current position.
        if ballX == self.predPositions[ 0 ][ 0 ] and ballY == self.predPositions[ 0 ][ 1 ]:
            # If it is then remove this predicted position.
            self.predPositions.pop( 0 )
            # And continue learning.
            self.LearnNextStepBall()
        else:
            # If not then generate new predicted positions.
            self.PredictTimeStepBall()
            # And reset tp and initiate new learning sequence.
            self.LearnInitialStepBall()

        while len( self.predPositions ) > self.maxPredLocations:
            self.predPositions.pop( -1 )

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
#        self.LearnTimeStepPaddle( self.memBuffer[ -2 ], self.memBuffer[ -1 ], -350, self.memBuffer[ -2 ][ 2 ], self.memBuffer[ -2 ][ 4 ] )
        # Learn last 2-step sequence centered around paddle_b.
#        self.LearnTimeStepPaddle( self.memBuffer[ -2 ], self.memBuffer[ -1 ], 350, self.memBuffer[ -2 ][ 3 ], self.memBuffer[ -2 ][ 5 ] )

#        return [ chosenMotorA, chosenMotorB ]
        return [ 1, 1 ]
