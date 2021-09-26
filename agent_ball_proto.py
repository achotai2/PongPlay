import sys
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

    localDimX   = 200
    localDimY   = 200

    maxPredLocations = 60
    maxStepsInPath   = 30
    overlapThresh    = 0.1
    maxMemoryDist    = 10
    maxLocalPosSize  = 1000

    def __init__( self, name, screenHeight, screenWidth, ballHeight, ballWidth, paddleHeight, paddleWidth ):

        self.ID = name

        self.screenHeight = screenHeight
        self.screenWidth  = screenWidth

        # Set up encoder parameters
        ballXEncodeParams    = ScalarEncoderParameters()
        ballYEncodeParams    = ScalarEncoderParameters()
        paddleEncodeParams   = ScalarEncoderParameters()
        wallXEncodeParams    = ScalarEncoderParameters()
        wallYEncodeParams    = ScalarEncoderParameters()
        centerXEncodeParams  = ScalarEncoderParameters()
        centerYEncodeParams  = ScalarEncoderParameters()

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

        centerXEncodeParams.activeBits = 11
        centerXEncodeParams.radius     = 5
        centerXEncodeParams.clipInput  = False
        centerXEncodeParams.minimum    = -int( screenWidth / 2 )
        centerXEncodeParams.maximum    = int( screenWidth / 2 )
        centerXEncodeParams.periodic   = False

        centerYEncodeParams.activeBits = 11
        centerYEncodeParams.radius     = 5
        centerYEncodeParams.clipInput  = False
        centerYEncodeParams.minimum    = -int( screenHeight / 2 )
        centerYEncodeParams.maximum    = int( screenHeight / 2 )
        centerYEncodeParams.periodic   = False

        # Set up encoders
        self.ballEncoderX    = ScalarEncoder( ballXEncodeParams )
        self.ballEncoderY    = ScalarEncoder( ballYEncodeParams )
        self.paddleEncoderY  = ScalarEncoder( paddleEncodeParams )
        self.wallEncoderX    = ScalarEncoder( wallXEncodeParams )
        self.wallEncoderY    = ScalarEncoder( wallYEncodeParams )
        self.centerEncoderX    = ScalarEncoder( centerXEncodeParams )
        self.centerEncoderY    = ScalarEncoder( centerYEncodeParams )

        self.encodingWidth = ( self.ballEncoderX.size + self.ballEncoderY.size + self.wallEncoderX.size +
            self.wallEncoderY.size + ( self.paddleEncoderY.size * 2 ) + self.centerEncoderX.size + self.centerEncoderY.size )

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

        self.predPositions = []
        self.memBuffer = [ [ 0, 0, 0, 0, 1, 1 ] ] * self.maxMemoryDist
#        self.ballLocalPosForCell = [] * self.sp.getColumnDimensions()[ 0 ]
        self.ballCellForLocalPos = []
        self.paddleAForLocalPos  = []
        self.paddleBForLocalPos  = []

    def Overlap ( self, SDR1, SDR2 ):
    # Computes overlap score between two passed SDRs.

        overlap = 0

        for cell1 in SDR1:
            if cell1 in SDR2:
                overlap += 1

        return overlap

    def EncodeSenseData ( self, ballX, ballY, paddleAY, paddleBY, centerX, centerY, learnIt ):
    # Encodes sense data as an SDR and returns it.

        # Encode global location center bits.
        centerBitsX = self.centerEncoderX.encode( centerX )
        centerBitsY = self.centerEncoderY.encode( centerY )

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
        encoding = SDR( self.encodingWidth ).concatenate( [ ballBitsX, ballBitsY, paddleABitsY, paddleBBitsY, wallBitsX, wallBitsY, centerBitsX, centerBitsY ] )
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, learnIt, senseSDR )

        # Feed SDR into tp.
        self.tp.compute( senseSDR, learn = learnIt )
        self.tp.activateDendrites( learn = learnIt )

        return senseSDR

    def ClassifyLocalPos ( self, thisSDR, ballX, ballY, paddleAY, paddleBY, centerX, centerY ):
    # Classifies thisSDR for ball and paddle local position.

        if Within( ballX - centerX, -self.localDimX / 2, self.localDimX / 2, True ) and Within( ballY - centerY, -self.localDimY / 2, self.localDimY / 2, True ):
            localBallX = ballX - centerX
            localBallY = ballY - centerY
        else:
            localBallX = None
            localBallY = None

        if Within( -350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ) and Within( paddleAY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), True ):
            localPaddleAY = paddleAY - centerY
        else:
            localPaddleAY = None

        if Within( 350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ) and Within( paddleBY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), True ):
            localPaddleBY = paddleBY - centerY
        else:
            localPaddleBY = None

        if localBallX != None and localBallY != None:
            foundPos = False
            for localPos in self.ballCellForLocalPos:
                if localPos[ 1 ] == localBallX and localPos[ 2 ] == localBallY:
                    foundPos = True

                    localPos[ 0 ] = numpy.append( localPos[ 0 ], thisSDR.sparse )
                    u, ind = numpy.unique( localPos[ 0 ], return_index = True )
                    localPos[ 0 ] = u[ numpy.argsort( ind ) ]
                    localPos[ 0 ] = u

                while localPos[ 0 ].size > self.maxLocalPosSize:
                    localPos[ 0 ] = numpy.delete( localPos[ 0 ], 0 )

            if not foundPos:
                self.ballCellForLocalPos.append( [ thisSDR.sparse, localBallX, localBallY ] )

        if localPaddleAY != None:
            foundPos = False
            for localPos in self.paddleAForLocalPos:
                if localPos[ 1 ] == localPaddleAY:
                    foundPos = True

                    localPos[ 0 ] = numpy.append( localPos[ 0 ], thisSDR.sparse )
                    u, ind = numpy.unique( localPos[ 0 ], return_index = True )
                    localPos[ 0 ] = u[ numpy.argsort( ind ) ]
                    localPos[ 0 ] = u

                while localPos[ 0 ].size > self.maxLocalPosSize:
                    localPos[ 0 ] = numpy.delete( localPos[ 0 ], 0 )

            if not foundPos:
                self.paddleAForLocalPos.append( [ thisSDR.sparse, localPaddleAY ] )

        if localPaddleBY != None:
            foundPos = False
            for localPos in self.paddleBForLocalPos:
                if localPos[ 1 ] == localPaddleBY:
                    foundPos = True

                    localPos[ 0 ] = numpy.append( localPos[ 0 ], thisSDR.sparse )
                    u, ind = numpy.unique( localPos[ 0 ], return_index = True )
                    localPos[ 0 ] = u[ numpy.argsort( ind ) ]
                    localPos[ 0 ] = u

                while localPos[ 0 ].size > self.maxLocalPosSize:
                    localPos[ 0 ] = numpy.delete( localPos[ 0 ], 0 )

            if not foundPos:
                self.paddleBForLocalPos.append( [ thisSDR.sparse, localPaddleBY ] )

#    def ClassifyMotor( self, thisSDR, centerX, chosenMotor ):
#    # Classifies thisSDR for chosenMotor.
#
#        if Within( -350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ):
#            self.paddleAMotorClass.learn( pattern = thisSDR, classification = chosenMotor )
#        elif Within( 350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ):
#            self.paddleBMotorClass.learn( pattern = thisSDR, classification = chosenMotor )

    def LearnTimeStepBall ( self ):
    # Learn the three time-step data for ball, centered around assigned last position.
    # Repeat sequence for maxMemoryDist length.

        self.tp.reset()

        lastElement   = self.memBuffer[ -3 ]
        centerElement = self.memBuffer[ -2 ]
        nextElement   = self.memBuffer[ -1 ]

        self.EncodeSenseData( lastElement[ 0 ], lastElement[ 1 ], lastElement[ 2 ], lastElement[ 3 ], centerElement[ 0 ], centerElement[ 1 ], True )
        winnerCellsTP = self.tp.getWinnerCells()
        self.ClassifyLocalPos( winnerCellsTP, lastElement[ 0 ], lastElement[ 1 ], lastElement[ 2 ], lastElement[ 3 ], centerElement[ 0 ], centerElement[ 1 ] )

        self.EncodeSenseData( centerElement[ 0 ], centerElement[ 1 ], centerElement[ 2 ], centerElement[ 3 ], centerElement[ 0 ], centerElement[ 1 ], True )
        winnerCellsTP = self.tp.getWinnerCells()
        self.ClassifyLocalPos( winnerCellsTP, centerElement[ 0 ], centerElement[ 1 ], centerElement[ 2 ], centerElement[ 3 ], centerElement[ 0 ], centerElement[ 1 ] )

        self.EncodeSenseData( nextElement[ 0 ], nextElement[ 1 ], nextElement[ 2 ], nextElement[ 3 ], centerElement[ 0 ], centerElement[ 1 ], True )
        winnerCellsTP = self.tp.getWinnerCells()
        self.ClassifyLocalPos( winnerCellsTP, nextElement[ 0 ], nextElement[ 1 ], nextElement[ 2 ], nextElement[ 3 ], centerElement[ 0 ], centerElement[ 1 ] )

#    def LearnTimeStepPaddle ( self, last, present, centerX, centerY, chosenMotor ):
#    # Learn motion of paddle in tune with chosen motor function.
#
#        self.tp.reset()
#
#        thisSDR = self.EncodeSenseData( last[ 0 ], last[ 1 ], last[ 2 ], last[ 3 ], centerX, centerY, True )
#        self.ClassifyLocalPos( thisSDR, last[ 0 ], last[ 1 ], last[ 2 ], last[ 3 ], centerX, centerY )
#        self.ClassifyMotor( thisSDR, centerX, chosenMotor )
#        thisSDR = self.EncodeSenseData( present[ 0 ], present[ 1 ], present[ 2 ], present[ 3 ], centerX, centerY, True )
#        self.ClassifyLocalPos( thisSDR, present[ 0 ], present[ 1 ], present[ 2 ], present[ 3 ], centerX, centerY )
#        self.ClassifyMotor( thisSDR, centerX, chosenMotor )

    def InferLocalPos ( self, thisSDR, localPosList ):
    # Makes an inference on classification of thisSDR using overlap scores.

        probabilities = []

        sizeSDR = thisSDR.sparse.size

        if sizeSDR > 0:
            for localPos in localPosList:
                overlapScore = self.Overlap( thisSDR.sparse, localPos[ 0 ] )
                probabilities.append( [ overlapScore, overlapScore / sizeSDR, [ localPos[ i ] for i in range( 1, len( localPos ) ) ] ] )

            probabilities.sort( key = lambda x: x[ 1 ], reverse = True )

        return probabilities

    def PredictSequenceBall ( self ):
    # Train time-step data, from secondlast to last, centered around last, and then predict next position and return.

#        print("---------------------------------------------")

        paths = []
        paths.append( [ self.memBuffer[ -2 ], self.memBuffer[ -1 ] ] )
        toAppend = []

        while len( paths ) > 0 and len( toAppend ) <= self.maxStepsInPath:
            lastElement   = paths[ -1 ][ 0 ]
            centerElement = paths[ -1 ][ 1 ]
            paths.pop( -1 )
#            print( "TEST:", lastElement, centerElement)

            self.tp.reset()

#            print( "LAST:", lastElement, "CENTER:", centerElement)

            self.EncodeSenseData( lastElement[ 0 ], lastElement[ 1 ], lastElement[ 2 ], lastElement[ 3 ], centerElement[ 0 ], centerElement[ 1 ], False )

            self.EncodeSenseData( centerElement[ 0 ], centerElement[ 1 ], centerElement[ 2 ], centerElement[ 3 ], centerElement[ 0 ], centerElement[ 1 ], False )
            predictCellsTP = self.tp.getPredictiveCells()

#            print( "NUM PREDICTED CELLS:", len(predictCellsTP.sparse))

            # Get predicted location for next time step.
#            predictions = self.InferLocalPos( predictCellsTP, self.paddleAForLocalPos )
#            print("Paddle A probabilities:", predictions )

#            for pred in predictions:
#                if pred[ 1 ] >= self.overlapThresh and pred[ 2 ][ 0 ] != None:
#                    nextPaddleAPos = centerElement[ 1 ] + pred[ 2 ][ 0 ]
#                    if nextPaddleAPos > self.screenHeight / 2:
#                        nextPaddleAPos = int( self.screenHeight / 2 )
#                    elif nextPaddleAPos < -self.screenHeight / 2:
#                        nextPaddleAPos[ 0 ] = -int( self.screenHeight / 2 )
#
#                    paths.append( [ centerElement, [ nextBallPos[ 0 ], nextBallPos[ 1 ], None, None ] ] )
#                    toAppend.append( [ nextBallPos[ 0 ], nextBallPos[ 1 ], None, None ] )

            predictions = self.InferLocalPos( predictCellsTP, self.ballCellForLocalPos )
            print("Ball probabilities:", predictions )

            for pred in predictions:
                if pred[ 1 ] >= self.overlapThresh and pred[ 2 ][ 0 ] != None and pred[ 2 ][ 1 ] != None:
                    nextBallPos = [ centerElement[ 0 ] + pred[ 2 ][ 0 ], centerElement[ 1 ] + pred[ 2 ][ 1 ] ]
                    if nextBallPos[ 0 ] > self.screenWidth / 2:
                        nextBallPos[ 0 ] = int( self.screenWidth / 2 )
                    elif nextBallPos[ 0 ] < -self.screenWidth / 2:
                        nextBallPos[ 0 ] = -int( self.screenWidth / 2 )
                    if nextBallPos[ 1 ] > self.screenHeight / 2:
                        nextBallPos[ 1 ] = int( self.screenHeight / 2 )
                    elif nextBallPos[ 1 ] < -self.screenHeight / 2:
                        nextBallPos[ 1 ] = -int( self.screenHeight / 2 )

                    paths.append( [ centerElement, [ nextBallPos[ 0 ], nextBallPos[ 1 ], None, None ] ] )
                    toAppend.append( [ nextBallPos[ 0 ], nextBallPos[ 1 ], None, None ] )

#                    if pred[ 4 ] != None:
#                        nextPaddleAPos = centerElement[ 1 ] + pred[ 4 ]
#                    else:
#                        nextPaddleAPos = None

#                    if pred[ 5 ] != None:
#                        nextPaddleBPos = centerElement[ 1 ] + pred[ 5 ]
#                    else:
#                        nextPaddleBPos = None


#                    if nextBallPos[ 0 ] == -350 and nextBallPos[ 1 ] == pred[ 4 ]:
#                        print("Ball predicted to hit paddle.")

#                    self.EncodeSenseData( nextPos[ 0 ], nextPos[ 1 ], centerElement[ 2 ], centerElement[ 3 ], centerElement[ 0 ], centerElement[ 1 ], False )

        self.predPositions.extend( toAppend )
#                print("APPEND:", [ centerElement, [ nextPos[ 0 ], nextPos[ 1 ], centerElement[ 2 ], centerElement[ 3 ] ] ])

        while len( self.predPositions ) > self.maxPredLocations:
            self.predPositions.pop( 0 )

    def Brain ( self, ballX, ballY, paddleAY, paddleBY ):
    # Agents brain center.

        # Set present ball coordinates for next time-step.
        self.memBuffer.append( [ ballX, ballY, paddleAY, paddleBY ] )
        while len( self.memBuffer ) > self.maxMemoryDist:
            self.memBuffer.pop( 0 )

        # Learn 3-step sequence, and jump, centered around ball. Do this back in memory.
        self.LearnTimeStepBall()
        print( "Size of learned positions:", len(self.ballCellForLocalPos))

        # Predict next ball positions and store them.
        self.PredictSequenceBall()
#        print( "Pred Positions:", self.predPositions)

        # If ball is predicted to fall off then try to move paddle there.
        paddleADirect = False
        paddleBDirect = False
        chosenMotorA = 1
        chosenMotorB = 1
        for pred in self.predPositions:
            if pred[ 0 ] <= -300:
                paddleADirect = True
                if paddleAY > pred[ 1 ]:
                    chosenMotorA = 2
#                    self.memBuffer[ -1 ][ 4 ] = chosenMotorA
                elif paddleAY < pred[ 1 ]:
                    chosenMotorA = 0
#                    self.memBuffer[ -1 ][ 4 ] = chosenMotorA
            elif pred[ 0 ] >= 300:
                paddleBDirect = True
                if paddleBY > pred[ 1 ]:
                    chosenMotorB = 2
#                    self.memBuffer[ -1 ][ 5 ] = chosenMotorB
                elif paddleBY < pred[ 1 ]:
                    chosenMotorB = 0
#                    self.memBuffer[ -1 ][ 5 ] = chosenMotorB

#        # If we aren't directing paddles movement then just learn paddle movement with random motion.
#        if not paddleADirect:
#            chosenMotorA = random.choice( [ 0, 1, 2 ] )
#            self.memBuffer[ -1 ][ 4 ] = chosenMotorA
#        if not paddleBDirect:
#            chosenMotorB = random.choice( [ 0, 1, 2 ] )
#            self.memBuffer[ -1 ][ 5 ] = chosenMotorB

        # Learn last 2-step sequence centered around paddle_a.
#        self.LearnTimeStepPaddle( self.memBuffer[ -2 ], self.memBuffer[ -1 ], -350, self.memBuffer[ -2 ][ 2 ], self.memBuffer[ -2 ][ 4 ] )
        # Learn last 2-step sequence centered around paddle_b.
#        self.LearnTimeStepPaddle( self.memBuffer[ -2 ], self.memBuffer[ -1 ], 350, self.memBuffer[ -2 ][ 3 ], self.memBuffer[ -2 ][ 5 ] )

        return [ chosenMotorA, chosenMotorB ]
#        return [1,1]
