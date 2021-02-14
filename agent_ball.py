import numpy

from htm.bindings.sdr import SDR, Metrics
import htm.bindings.encoders
ScalarEncoder = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.algorithms import Classifier

class BallAgent:

    motorDimensions = 3
    synapseInc      = 0.5
    synapseDec      = 0.1

    def __init__( self, name, screenHeight, screenWidth ):

        self.ID = name

        self.screenHeight = screenHeight
        self.screenWidth  = screenWidth

        # Set up encoder parameters
        localXEncodeParams    = ScalarEncoderParameters()
        localYEncodeParams    = ScalarEncoderParameters()
        ballXEncodeParams     = ScalarEncoderParameters()
        ballYEncodeParams     = ScalarEncoderParameters()

        localXEncodeParams.activeBits = 21
        localXEncodeParams.radius     = 20
        localXEncodeParams.clipInput  = False
        localXEncodeParams.minimum    = -50
        localXEncodeParams.maximum    = 50
        localXEncodeParams.periodic   = False

        localYEncodeParams.activeBits = 21
        localYEncodeParams.radius     = 20
        localYEncodeParams.clipInput  = False
        localYEncodeParams.minimum    = -50
        localYEncodeParams.maximum    = 50
        localYEncodeParams.periodic   = False

        ballXEncodeParams.activeBits = 5
        ballXEncodeParams.radius     = 20
        ballXEncodeParams.clipInput  = False
        ballXEncodeParams.minimum    = -screenWidth / 2
        ballXEncodeParams.maximum    = screenWidth / 2
        ballXEncodeParams.periodic   = False

        ballYEncodeParams.activeBits = 5
        ballYEncodeParams.radius     = 20
        ballYEncodeParams.clipInput  = False
        ballYEncodeParams.minimum    = -screenHeight / 2
        ballYEncodeParams.maximum    = screenHeight / 2
        ballYEncodeParams.periodic   = False

        # Set up encoders
        self.localEncoderX    = ScalarEncoder( localXEncodeParams )
        self.localEncoderY    = ScalarEncoder( localYEncodeParams )
        self.ballEncoderX     = ScalarEncoder( ballXEncodeParams )
        self.ballEncoderY     = ScalarEncoder( ballYEncodeParams )

        self.encodingWidth = ( self.localEncoderX.size + self.localEncoderY.size + self.ballEncoderX.size +
            self.ballEncoderY.size )

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

        self.xPosition = Classifier( alpha = 1 )
        self.yPosition = Classifier( alpha = 1 )

        self.lastData       = [ 0, 0 ]
        self.secondLastData = [ 0, 0 ]

    def EncodeSenseData ( self, localX, localY, ballX, ballY ):
    # Encodes sense data as an SDR and returns it.

        # Now we call the encoders to create bit representations for each value, and encode them.
        localBitsX   = self.localEncoderX.encode( localX )
        localBitsY   = self.localEncoderY.encode( localY )
        ballBitsX    = self.ballEncoderX.encode( ballX )
        ballBitsY    = self.ballEncoderY.encode( ballY )

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( self.encodingWidth ).concatenate( [ localBitsX, localBitsY, ballBitsX, ballBitsY ] )
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    def LearnTimeStep ( self, secondLast, last, present, doLearn ):
    # Learn the three time-step data, from second last to last to present, centered around last time-step.

        self.tp.reset()

        if secondLast[ 0 ] - last[ 0 ] <= 100 and secondLast[ 0 ] - last[ 0 ] >= -100 and secondLast[ 1 ] - last[ 1 ] <= 100 and secondLast[ 1 ] - last[ 1 ] >= -100:
            secondLastSDR = self.EncodeSenseData( secondLast[ 0 ] - last[ 0 ], secondLast[ 1 ] - last[ 1 ], last[ 0 ], last[ 1 ] )

            # Feed x and y position into classifier to learn.
            # Classifier can only take positive input, so need to transform ball origin.
            if doLearn:
                self.xPosition.learn( pattern = secondLastSDR, classification = secondLast[ 0 ] - last[ 0 ] + 50 )
                self.yPosition.learn( pattern = secondLastSDR, classification = secondLast[ 1 ] - last[ 1 ] + 50 )

            # Reset tp and feed SDR into tp.
            self.tp.compute( secondLastSDR, learn = doLearn )
            self.tp.activateDendrites( learn = doLearn )

        # Generate SDR for last sense data by feeding sense data into SP with learning.
        lastSDR = self.EncodeSenseData( 0, 0, last[ 0 ], last[ 1 ] )

        if doLearn:
            self.xPosition.learn( pattern = lastSDR, classification = 50 )
            self.yPosition.learn( pattern = lastSDR, classification = 50 )

        self.tp.compute( lastSDR, learn = doLearn )
        self.tp.activateDendrites( learn = doLearn )

        if present[ 0 ] - last[ 0 ] <= 100 and present[ 0 ] - last[ 0 ] >= -100 and present[ 1 ] - last[ 1 ] <= 100 and present[ 1 ] - last[ 1 ] >= -100:
            # Generate SDR for last sense data by feeding sense data into SP with learning.
            senseSDR = self.EncodeSenseData( present[ 0 ] - last[ 0 ], present[ 1 ] - last[ 1 ], last[ 0 ], last[ 1 ] )

            if doLearn:
                self.xPosition.learn( pattern = senseSDR, classification = present[ 0 ] - last[ 0 ] + 50 )
                self.yPosition.learn( pattern = senseSDR, classification = present[ 1 ] - last[ 1 ] + 50 )

            self.tp.compute( senseSDR, learn = doLearn )
            self.tp.activateDendrites( learn = doLearn )

    def PredictTimeStep ( self, secondLast, last, doLearn ):
    # Train time-step data, from secondlast to last, centered around last, and then predict next position and return.

        self.tp.reset()

        if secondLast[ 0 ] - last[ 0 ] <= 100 and secondLast[ 0 ] - last[ 0 ] >= -100 and secondLast[ 1 ] - last[ 1 ] <= 100 and secondLast[ 1 ] - last[ 1 ] >= -100:
            secondLastSDR = self.EncodeSenseData( secondLast[ 0 ] - last[ 0 ], secondLast[ 1 ] - last[ 1 ], last[ 0 ], last[ 1 ] )
            self.tp.compute( secondLastSDR, learn = doLearn )
            self.tp.activateDendrites( learn = doLearn )

        lastSDR = self.EncodeSenseData( 0, 0, last[ 0 ], last[ 1 ] )
        self.tp.compute( lastSDR, learn = doLearn )
        self.tp.activateDendrites( learn = doLearn )
        predictCellsTP = self.tp.getPredictiveCells()

        # Get predicted location for next time step.
        stepSenseSDR = SDR( self.sp.getColumnDimensions() )
        stepSenseSDR.sparse = numpy.unique( [ self.tp.columnForCell( cell ) for cell in predictCellsTP.sparse ] )
        positionX = numpy.argmax( self.xPosition.infer( pattern = stepSenseSDR ) ) - 50
        positionY = numpy.argmax( self.yPosition.infer( pattern = stepSenseSDR ) ) - 50

        return [ last[ 0 ] + positionX, last[ 1 ] + positionY ]

    def Brain ( self, ballX, ballY ):
    # Agents brain center.

        self.LearnTimeStep( self.secondLastData, self.lastData, [ ballX, ballY ], True )

        # Set present ball coordinates for next time-step.
        self.secondLastData = [ self.lastData[ 0 ], self.lastData[ 1 ] ]
        self.lastData = [ ballX, ballY ]

        # Store last and present locations.
        predPositions = []
        predPositions.append( self.secondLastData )
        predPositions.append( self.lastData )

        # Predict next 10 time step locations and store them.
        for step in range( 10 ):
            nextPosition = self.PredictTimeStep( predPositions[ -2 ], predPositions[ -1 ], False )
            if nextPosition[ 0 ] <= self.screenWidth / 2 and nextPosition[ 0 ] >= -self.screenWidth / 2 and nextPosition[ 1 ] <= self.screenHeight / 2 and nextPosition[ 1 ] >= -self.screenHeight / 2:
                predPositions.append( nextPosition )
            else:
                print(nextPosition)
                break

        print(predPositions)

        return predPositions
