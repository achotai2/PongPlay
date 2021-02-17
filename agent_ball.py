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

    resolutionX = 50
    resolutionY = 50

    def __init__( self, name, screenHeight, screenWidth, ballHeight, ballWidth, paddleHeight, paddleWidth ):

        self.ID = name

        self.screenHeight = screenHeight
        self.screenWidth  = screenWidth
        self.ballHeight   = ballHeight
        self.ballWidth    = ballWidth
        self.paddleHeight = paddleHeight
        self.paddleWidth  = paddleWidth

        # Set up encoder parameters
        localXEncodeParams    = ScalarEncoderParameters()
        localYEncodeParams    = ScalarEncoderParameters()
        ballXEncodeParams     = ScalarEncoderParameters()
        ballYEncodeParams     = ScalarEncoderParameters()

        # Encodes center position of attention screen.
        localXEncodeParams.activeBits = 5
        localXEncodeParams.radius     = 5
        localXEncodeParams.clipInput  = False
        localXEncodeParams.minimum    = -int( screenWidth / 2 )
        localXEncodeParams.maximum    = int( screenWidth / 2 )
        localXEncodeParams.periodic   = False

        localYEncodeParams.activeBits = 5
        localYEncodeParams.radius     = 5
        localYEncodeParams.clipInput  = False
        localYEncodeParams.minimum    = -int( screenHeight / 2 )
        localYEncodeParams.maximum    = int( screenHeight / 2 )
        localYEncodeParams.periodic   = False

        # Encodes position of ball within local attention screen.
        ballXEncodeParams.activeBits = ( self.ballWidth * 20 ) + 1
        ballXEncodeParams.radius     = ( self.ballWidth * 20 )
        ballXEncodeParams.clipInput  = True
        ballXEncodeParams.minimum    = -int( self.localDimX / 2 )
        ballXEncodeParams.maximum    = int( self.localDimX / 2 )
        ballXEncodeParams.periodic   = False

        ballYEncodeParams.activeBits = ( self.ballHeight * 20 ) + 1
        ballYEncodeParams.radius     = ( self.ballHeight * 20 )
        ballYEncodeParams.clipInput  = True
        ballYEncodeParams.minimum    = -int( self.localDimY / 2 )
        ballYEncodeParams.maximum    = int( self.localDimY / 2 )
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

    def Within ( self, value, minimum, maximum ):
    # Checks if value is <= maximum and >= minimum.

        if value <= maximum and value >= minimum:
            return True
        else:
            return False

    def EncodeSenseData ( self, localCenterX, localCenterY, ballX, ballY ):
    # Encodes sense data as an SDR and returns it.

        # Now we call the encoders to create bit representations for each value, and encode them.
        localBitsX   = self.localEncoderX.encode( localCenterX )
        localBitsY   = self.localEncoderY.encode( localCenterY )
        if self.Within( ballX - localCenterX, -self.localDimX / 2, self.localDimX / 2 ) and self.Within( ballY - localCenterY, -self.localDimY / 2, self.localDimY / 2 ):
            ballBitsX = self.ballEncoderX.encode( ballX )
            ballBitsY = self.ballEncoderY.encode( ballY )
        else:
            ballBitsX = SDR( self.ballEncoderX.size )
            ballBitsY = SDR( self.ballEncoderY.size )

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( self.encodingWidth ).concatenate( [ localBitsX, localBitsY, ballBitsX, ballBitsY ] )
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    def BuildLocalBitRep( self, localDimX, localDimY, centerX, centerY, ballX, ballY ):
    # Builds a bit-rep SDR of localDim dimensions centered around point with resolution.

        localBitRep = []

        scaleX = localDimX / self.resolutionX
        scaleY = localDimY / self.resolutionY

        if scaleX < 1 or scaleY < 1:
            sys.exit( "Resolution (X and Y) must be less than local dimensions (X and Y)." )

        # Right side border bits.
        if centerX + ( localDimX / 2 ) >= ( self.screenWidth / 2 ):
            for y in range( self.resolutionY ):
                localBitRep.append( int( ( ( self.screenWidth / 2 ) - centerX + ( localDimX / 2 ) ) / scaleX ) + ( y * self.resolutionX ) )

        # Left side border bits.
        if centerX - ( localDimX / 2 ) <= -( self.screenWidth / 2 ):
            for y in range( self.resolutionY ):
                localBitRep.append( int( ( -( self.screenWidth / 2 ) - centerX + ( localDimX / 2 ) ) / scaleX ) + ( y * self.resolutionX ) )

        # Top side border bits.
        if centerY + ( localDimX / 2 ) >= ( self.screenHeight / 2 ):
            for x in range( self.resolutionX ):
                localBitRep.append( x + ( int( ( (self.screenHeight / 2 ) - centerY + ( localDimY / 2 ) ) / scaleY ) * self.resolutionX ) )

        # Bottom side border bits.
        if centerY - ( localDimX / 2 ) <= -( self.screenHeight / 2 ):
            for x in range( self.resolutionX ):
                localBitRep.append( x + ( int( ( -(self.screenHeight / 2 ) - centerY + ( localDimY / 2 ) ) / scaleY ) * self.resolutionX ) )

        # Ball bits.
        if self.Within( ballX, centerX - ( localDimX / 2 ) - ( self.ballWidth * 10 ), centerX + ( localDimX / 2 ) + ( self.ballWidth * 10 ) ) and self.Within( ballY, centerY - ( localDimY / 2 ) - ( self.ballHeight * 10 ), centerY + ( localDimY / 2 ) + ( self.ballHeight * 10 ) ):
            for x in range( self.ballWidth * 20 ):
                for y in range( self.ballHeight * 20 ):
                    if self.Within( ballX - ( self.ballWidth * 10 ) + x, centerX - ( localDimX / 2 ), centerX + ( localDimX / 2 ) ) and self.Within( ballY - ( self.ballHeight * 10 ) + y, centerY - ( localDimY / 2 ), centerY + ( localDimY / 2 ) ):
                        bitX = ballX - ( self.ballWidth * 10 ) + x - centerX + ( localDimX / 2 )
                        bitY = ballY - ( self.ballHeight * 10 ) + y - centerY + ( localDimY / 2 )
                        localBitRep.append( int( bitX / scaleX ) + ( int( bitY / scaleY ) * self.resolutionX ) )

#        # Build unscaled bit representation.resolution
#        for x in range( localDimX ):
#            for y in range( localDimY ):
#                # Look if element at this position is in bitRep.
#                if x + centerX + int( bitRepDimX / 2 ) - int( localDimX / 2 ) + ( ( y + centerY + int( bitRepDimY / 2 ) - int( localDimY / 2 ) ) * bitRepDimX ) in bitRep:
#                    localBitRep.append( x + ( y * localDimX ) )

        # Scale bit representation to fit requested dimension.
#        scaleX = int( localDimX / self.resolutionX )
#        scaleY = int( localDimY / self.resolutionY )

#        localScaledBitRep = []

#        if scaleX > 1:
#            for x in range( self.resolutionX ):
#                for y in range ( self.resolutionY ):
#                    inSquare = False
#                    for sx in range( scaleX ):
#                        for sy in range( scaleY ):
#                            if sx + ( x * scaleX ) + ( ( sy + ( y * scaleY ) * localDimY ) ) in localBitRep:
#                                inSquare = True
#                    if inSquare == True:
#                        localScaledBitRep.append( x + ( y * self.resolutionX ) )

        print("----------------------------------------------------------------")

        whatPrintX   = self.resolutionX
        whatPrintY   = self.resolutionY
        whatPrintRep = localBitRep

        for y in range( whatPrintY ):
            for x in range( whatPrintX ):
                if x == whatPrintX - 1:
                    print ("ENDO")
                    endRep = "\n"
                else:
                    endRep = ""
                    if x + (y * whatPrintX ) in whatPrintRep:
                        print(1, end=endRep)
                    else:
                        print(0, end=endRep)

        bitRepSDR = SDR( self.resolutionX * self.resolutionY )
        bitRepSDR.sparse = numpy.unique( localBitRep )
        return bitRepSDR

    def LearnTimeStep ( self, secondLast, last, present, doLearn ):
    # Learn the three time-step data, from second last to last to present, centered around last time-step.

        self.tp.reset()

        if self.Within( secondLast[ 0 ] - last[ 0 ], -self.localDimX, self.localDimX ) and self.Within( secondLast[ 1 ] - last[ 1 ], -self.localDimY, self.localDimY ):
            secondLastSDR = self.EncodeSenseData( secondLast[ 0 ] - last[ 0 ], secondLast[ 1 ] - last[ 1 ], last[ 0 ], last[ 1 ] )

            # Feed x and y position into classifier to learn.
            # Classifier can only take positive input, so need to transform ball origin.
            if doLearn:
                self.xPosition.learn( pattern = secondLastSDR, classification = secondLast[ 0 ] - last[ 0 ] + int( self.localDimX / 2 ) )
                self.yPosition.learn( pattern = secondLastSDR, classification = secondLast[ 1 ] - last[ 1 ] + int( self.localDimY / 2 ) )

            # Feed SDR into tp.
            self.tp.compute( secondLastSDR, learn = doLearn )
            self.tp.activateDendrites( learn = doLearn )

        # Generate SDR for last sense data by feeding sense data into SP with learning.
        lastSDR = self.EncodeSenseData( 0, 0, last[ 0 ], last[ 1 ] )

        if doLearn:
            self.xPosition.learn( pattern = lastSDR, classification = int( self.localDimX / 2 ) )
            self.yPosition.learn( pattern = lastSDR, classification = int( self.localDimY / 2 ) )

        self.tp.compute( lastSDR, learn = doLearn )
        self.tp.activateDendrites( learn = doLearn )

        if self.Within( present[ 0 ] - last[ 0 ], -self.localDimX, self.localDimX ) and self.Within( present[ 1 ] - last[ 1 ], -self.localDimY, self.localDimY ):
            # Generate SDR for last sense data by feeding sense data into SP with learning.
            senseSDR = self.EncodeSenseData( present[ 0 ] - last[ 0 ], present[ 1 ] - last[ 1 ], last[ 0 ], last[ 1 ] )

            if doLearn:
                self.xPosition.learn( pattern = senseSDR, classification = present[ 0 ] - last[ 0 ] + int( self.localDimX / 2 ) )
                self.yPosition.learn( pattern = senseSDR, classification = present[ 1 ] - last[ 1 ] + int( self.localDimY / 2 ) )

            self.tp.compute( senseSDR, learn = doLearn )
            self.tp.activateDendrites( learn = doLearn )

    def PredictTimeStep ( self, secondLast, last, doLearn ):
    # Train time-step data, from secondlast to last, centered around last, and then predict next position and return.

        self.tp.reset()

        if self.Within( secondLast[ 0 ] - last[ 0 ], -self.localDimX, self.localDimX ) and self.Within( secondLast[ 1 ] - last[ 1 ], -self.localDimY, self.localDimY ):
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
        positionX = numpy.argmax( self.xPosition.infer( pattern = stepSenseSDR ) ) - int( self.localDimX / 2 )
        positionY = numpy.argmax( self.yPosition.infer( pattern = stepSenseSDR ) ) - int( self.localDimY / 2 )

        return [ last[ 0 ] + positionX, last[ 1 ] + positionY ]

    def Brain ( self, ballX, ballY ):
    # Agents brain center.

        self.BuildLocalBitRep( 100, 100, ballX, ballY, ballX, ballY )

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
            if self.Within( nextPosition[ 0 ], -self.screenWidth / 2, self.screenWidth / 2 ) and self.Within( nextPosition[ 1 ], -self.screenHeight / 2, self.screenHeight / 2 ):
                predPositions.append( nextPosition )
            else:
                break

        return predPositions
