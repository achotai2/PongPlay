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

    localDimX   = 100
    localDimY   = 100

    maxPredLocations = 20
    maxMemoryDist = 10

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
        self.paddleEncoderY  = ScalarEncoder( paddleEncodeParams )
        self.wallEncoderX    = ScalarEncoder( wallXEncodeParams )
        self.wallEncoderY    = ScalarEncoder( wallYEncodeParams )

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

        self.predPositions = []
        self.memBuffer = [ [ 0, 0, 0, 0, 1, 1 ] ] * self.maxMemoryDist
#        self.ballLocalPosForCell = [] * self.sp.getColumnDimensions()[ 0 ]
        self.ballCellForLocalPos = []

    def Overlap ( self, SDR1, SDR2 ):
    # Computes overlap score between two passed SDRs.

        overlap = 0

        for cell1 in SDR1.sparse:
            if cell1 in SDR2.sparse:
                overlap += 1

        return overlap

    def EncodeSenseData ( self, ballX, ballY, paddleAY, paddleBY, centerX, centerY, learnIt ):
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
        encoding = SDR( self.encodingWidth ).concatenate( [ ballBitsX, ballBitsY, paddleABitsY, paddleBBitsY, wallBitsX, wallBitsY ] )
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, learnIt, senseSDR )

        # Feed SDR into tp.
        self.tp.compute( senseSDR, learn = learnIt )
        self.tp.activateDendrites( learn = learnIt )

        return senseSDR

    def ClassifyLocalPos ( self, thisSDR, ballX, ballY, paddleAY, paddleBY, centerX, centerY ):
    # Performs learning on the 1-step of the 3-step sequence.

        # Feed x and y position into classifier to learn ball and paddle positions.
        # Classifier can only take positive input, so need to transform origin.
        if ballX !=  None and ballY != None and Within( ballX - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ) and Within( ballY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), True ):
            self.ballLocalXClass.learn( pattern = thisSDR, classification = int( ballX - centerX + ( self.localDimX / 2 ) ) )
            self.ballLocalYClass.learn( pattern = thisSDR, classification = int( ballY - centerY + ( self.localDimY / 2 ) ) )

            foundPos = False
            for localPos in self.ballCellForLocalPos:
                if localPos[ 0 ] == ballX - centerX and localPos[ 1 ] == ballY - centerY:
#                    for cell in thisSDR.sparse:
#                        localPos[ 2 ][ cell ] += 1
#                    localPos[ 3 ] += 1
                    localPos[ 2 ].sparse = numpy.union1d( thisSDR.sparse, localPos[ 2 ].sparse )
                    foundPos = True
            if not foundPos:
                self.ballCellForLocalPos.append( [ ballX - centerX, ballY - centerY, thisSDR ] )
#                self.ballCellForLocalPos.append( [ ballX - centerX, ballY - centerY, [ 0 ] * self.sp.getColumnDimensions()[ 0 ] , 1 ] )
#                for cell in thisSDR.sparse:
#                    self.ballCellForLocalPos[ -1 ][ 2 ][ cell ] += 1

        if paddleAY != None and Within( -350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ) and Within( paddleAY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), True ):
            self.paddleALocalYClass.learn( pattern = thisSDR, classification = int( paddleAY - centerY + ( self.localDimY / 2 ) ) )
        if paddleBY != None and Within( 350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ) and Within( paddleBY - centerY, -int( self.localDimY / 2 ), int( self.localDimY / 2 ), True ):
            self.paddleBLocalYClass.learn( pattern = thisSDR, classification = int( paddleBY - centerY + ( self.localDimY / 2 ) ) )

    def ClassifyMotor( self, thisSDR, centerX, chosenMotor ):

        if Within( -350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ):
            self.paddleAMotorClass.learn( pattern = thisSDR, classification = chosenMotor )
        elif Within( 350 - centerX, -int( self.localDimX / 2 ), int( self.localDimX / 2 ), True ):
            self.paddleBMotorClass.learn( pattern = thisSDR, classification = chosenMotor )

    def LearnTimeStepBall ( self, secondLast, last, present ):
    # Learn the three time-step data for ball, centered around assigned center position.

        self.tp.reset()
        thisSDR = self.EncodeSenseData( secondLast[ 0 ], secondLast[ 1 ], secondLast[ 2 ], secondLast[ 3 ], last[ 0 ], last[ 1 ], True )
        self.ClassifyLocalPos( thisSDR, secondLast[ 0 ], secondLast[ 1 ], secondLast[ 2 ], secondLast[ 3 ], last[ 0 ], last[ 1 ] )
        thisSDR = self.EncodeSenseData( last[ 0 ], last[ 1 ], last[ 2 ], last[ 3 ], last[ 0 ], last[ 1 ], True )
        self.ClassifyLocalPos( thisSDR, last[ 0 ], last[ 1 ], last[ 2 ], last[ 3 ], last[ 0 ], last[ 1 ] )
        thisSDR = self.EncodeSenseData( present[ 0 ], present[ 1 ], present[ 2 ], present[ 3 ], last[ 0 ], last[ 1 ], True )
        self.ClassifyLocalPos( thisSDR, present[ 0 ], present[ 1 ], present[ 2 ], present[ 3 ], last[ 0 ], last[ 1 ] )

    def LearnTimeStepPaddle ( self, last, present, centerX, centerY, chosenMotor ):
    # Learn motion of paddle in tune with chosen motor function.

        self.tp.reset()
        thisSDR = self.EncodeSenseData( last[ 0 ], last[ 1 ], last[ 2 ], last[ 3 ], centerX, centerY, True )
        self.ClassifyLocalPos( thisSDR, last[ 0 ], last[ 1 ], last[ 2 ], last[ 3 ], centerX, centerY )
        self.ClassifyMotor( thisSDR, centerX, chosenMotor )
        thisSDR = self.EncodeSenseData( present[ 0 ], present[ 1 ], present[ 2 ], present[ 3 ], centerX, centerY, True )
        self.ClassifyLocalPos( thisSDR, present[ 0 ], present[ 1 ], present[ 2 ], present[ 3 ], centerX, centerY )
        self.ClassifyMotor( thisSDR, centerX, chosenMotor )

    def OLDOLDInferLocalPos( self, thisSDR ):
    # P( A|B ) = Probability that we are in this local pos state given that this cell is firing.
    #               We compute using Baysian analysis.
    # P( B|A ) = Probability that this cell is firing given that we are in this local pos state.
    #               We compute by storing the cell firings for each local pos state.
    # P( B )   = Probability that this cell is firing.
    #               We compute this by assuming all cells have equal likelihood (given boosting).
    # P( A )   = Probability that we are in this local pos state.
    #               We compute by assuming it's 1 / found pos. It's the same for all so shouldn't matter.
    #   http://cs.wellesley.edu/~anderson/writing/naive-bayes.pdf

        allCellsAllLocalProd = []
        allLocalSum          = []
        allLocalTotalSum     = 0.0
        allCellAllLocalSum   = [ 0.0 ] * self.sp.getColumnDimensions()[ 0 ]
        for localPos in self.ballCellForLocalPos:
            allLocalSum.append( localPos[ 3 ] )
            allLocalTotalSum += localPos[ 3 ]
            allCellsThisLocalProd = 1.0
            for cell in thisSDR.sparse:
                # Compute P( B|A ) for this cell (B) and this localPos (A).
                allCellsThisLocalProd *= localPos[ 2 ][ cell ] / localPos[ 3 ]
                allCellAllLocalSum[ cell ] += localPos[ 2 ][ cell ]
            allCellsAllLocalProd.append( allCellsThisLocalProd )

        allCellsNotAllLocalProd = []
        for idx, localPos in enumerate( self.ballCellForLocalPos ):
            allCellsNotThisLocalProd = 1.0
            for cell in thisSDR.sparse:
                if allLocalTotalSum - allLocalSum[ idx ] != 0.0:
                    allCellsNotThisLocalProd *= ( allCellAllLocalSum[ cell ] - localPos[ 2 ][ cell ] ) / ( allLocalTotalSum - allLocalSum[ idx ] )
            allCellsNotAllLocalProd.append( allCellsNotThisLocalProd )

        probabilities = []

        for lp in range( len( self.ballCellForLocalPos ) ):
            if ( allLocalSum[ lp ] * allCellsAllLocalProd[ lp ] ) + ( allLocalTotalSum - allLocalSum[ lp ] ) * allCellsNotAllLocalProd[ lp ] != 0.0:
                probAB = allLocalSum[ lp ] * allCellsAllLocalProd[ lp ] / ( ( allLocalSum[ lp ] * allCellsAllLocalProd[ lp ] ) + ( allLocalTotalSum - allLocalSum[ lp ] ) * allCellsNotAllLocalProd[ lp ] )
            else:
                probAB = 0.0
                print("Numerator was zero.")

            probabilities.append( [ self.ballCellForLocalPos[ lp ][ 0 ], self.ballCellForLocalPos[ lp ][ 1 ], probAB ] )

        return probabilities

    def OLDInferLocalPos( self, thisSDR ):
    # P( A|B ) = Probability that we are in this local pos state given that this cell is firing.
    #               We compute using Baysian analysis.
    # P( B|A ) = Probability that this cell is firing given that we are in this local pos state.
    #               We compute by storing the cell firings for each local pos state.
    # P( B )   = Probability that this cell is firing.
    #               We compute this by assuming all cells have equal likelihood (given boosting).
    # P( A )   = Probability that we are in this local pos state.
    #               We compute by assuming it's 1 / found pos. It's the same for all so shouldn't matter.
    #   http://cs.wellesley.edu/~anderson/writing/naive-bayes.pdf

        probStateGivenSDR        = []
        sumForEachState          = []
        totalSumForAllStates     = 0.0
        sumForCellForAllStates   = [ 0.0 ] * self.sp.getColumnDimensions()[ 0 ]
        for localPos in self.ballCellForLocalPos:
            sumForEachState.append( localPos[ 3 ] )
            totalSumForAllStates += localPos[ 3 ]
            probStateGivenCells = 0.0
            for cell in thisSDR.sparse:
                # Compute P( B|A ) for this cell (B) and this localPos (A).
                probStateGivenCells += localPos[ 2 ][ cell ] / localPos[ 3 ]
                sumForCellForAllStates[ cell ] += localPos[ 2 ][ cell ]
            probStateGivenSDR.append( probStateGivenCells )

        probNotStateGivenSDR = []
        for idx, localPos in enumerate( self.ballCellForLocalPos ):
            probNotStateGivenCells = 0.0
            for cell in thisSDR.sparse:
                if totalSumForAllStates - sumForEachState[ idx ] != 0.0:
                    probNotStateGivenCells += ( sumForCellForAllStates[ cell ] - localPos[ 2 ][ cell ] ) / ( totalSumForAllStates - sumForEachState[ idx ] )
            probNotStateGivenSDR.append( probNotStateGivenCells )

        probabilities = []

        for lp in range( len( self.ballCellForLocalPos ) ):
            if ( ( sumForEachState[ lp ] * probStateGivenSDR[ lp ] ) + ( totalSumForAllStates - sumForEachState[ lp ] ) * probNotStateGivenSDR[ lp ] ) != 0.0:
                probAB = sumForEachState[ lp ] * probStateGivenSDR[ lp ] / ( ( sumForEachState[ lp ] * probStateGivenSDR[ lp ] ) + ( totalSumForAllStates - sumForEachState[ lp ] ) * probNotStateGivenSDR[ lp ] )
            else:
                probAB = 0.0
                print("Numerator was zero.")
            probabilities.append( [ self.ballCellForLocalPos[ lp ][ 0 ], self.ballCellForLocalPos[ lp ][ 1 ], probAB ] )

        print(probabilities)

        return probabilities

    def InferLocalPos ( self, thisSDR ):

        probabilities = []

        for localPos in self.ballCellForLocalPos:
            probabilities.append( [ localPos[ 0 ], localPos[ 1 ], self.Overlap( thisSDR, localPos[ 2 ] ) ] )

        probabilities.sort( key = lambda x: x[ 2 ], reverse = True )
        print("Predicted Cells:", thisSDR)
        print("Probabilities:", probabilities )
        return probabilities

    def PredictTimeStep ( self, secondLast, last, predStep ):
    # Train time-step data, from secondlast to last, centered around last, and then predict next position and return.

        self.tp.reset()

        secondLastSDR = self.EncodeSenseData( secondLast[ 0 ], secondLast[ 1 ], secondLast[ 2 ], secondLast[ 3 ], last[ 0 ], last[ 1 ], False )

        lastSDR = self.EncodeSenseData( last[ 0 ], last[ 1 ], last[ 2 ], last[ 3 ], last[ 0 ], last[ 1 ], False )
        predictCellsTP = self.tp.getPredictiveCells()

        # Get predicted location for next time step.
        stepSenseSDR = SDR( self.sp.getColumnDimensions() )
        stepSenseSDR.sparse = numpy.unique( [ self.tp.columnForCell( cell ) for cell in predictCellsTP.sparse ] )
        greatestPred = self.InferLocalPos( stepSenseSDR )[ 0 ]
        if greatestPred[ 2 ] != 0.0:
            positionX = greatestPred[ 0 ]
            positionY = greatestPred[ 1 ]
        else:
            positionX = 0
            positionY = 0
#        positionX = numpy.argmax( self.ballLocalXClass.infer( pattern = stepSenseSDR ) ) - int( self.localDimX / 2 )
#        positionY = numpy.argmax( self.ballLocalYClass.infer( pattern = stepSenseSDR ) ) - int( self.localDimY / 2 )

        print( "Prediction:", positionX, positionY )
        print( "Pred Step:", predStep )

        return [ last[ 0 ] + positionX, last[ 1 ] + positionY, last[ 2 ], last[ 3 ] ]

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

        paddleADirect = False
        paddleBDirect = False
        chosenMotorA = 1
        chosenMotorB = 1

        # Predict next 10 time step locations and store them.
        for step in range( self.maxPredLocations - 2):
            nextPosition = self.PredictTimeStep( self.predPositions[ -2 ], self.predPositions[ -1 ], step )
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
#        self.LearnTimeStepPaddle( self.memBuffer[ -2 ], self.memBuffer[ -1 ], -350, self.memBuffer[ -2 ][ 2 ], self.memBuffer[ -2 ][ 4 ] )
        # Learn last 2-step sequence centered around paddle_b.
#        self.LearnTimeStepPaddle( self.memBuffer[ -2 ], self.memBuffer[ -1 ], 350, self.memBuffer[ -2 ][ 3 ], self.memBuffer[ -2 ][ 5 ] )

        return [ chosenMotorA, chosenMotorB ]
#        return [1,1]
