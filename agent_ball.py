import numpy

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
    resolutionX = 24
    resolutionY = 24

    maxMemoryDist = 10

    def __init__( self, name, screenHeight, screenWidth, ballHeight, ballWidth, paddleHeight, paddleWidth ):

        self.ID = name

        self.screenHeight = screenHeight
        self.screenWidth  = screenWidth
        self.ballHeight   = ballHeight
        self.ballWidth    = ballWidth
        self.paddleHeight = paddleHeight
        self.paddleWidth  = paddleWidth

        # Set up encoder parameters
        centerXEncodeParams   = ScalarEncoderParameters()
        centerYEncodeParams   = ScalarEncoderParameters()

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
        self.centerXEncoder   = ScalarEncoder( centerXEncodeParams )
        self.centerYEncoder   = ScalarEncoder( centerYEncodeParams )

        self.encodingWidth = ( ( self.resolutionX * self.resolutionY ) + self.centerXEncoder.size +
            self.centerYEncoder.size )

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

        self.predPositions = []
        self.memBuffer = [ [ 0, 0, 0 ], [ 0, 0, 0 ] ]

    def PrintBitRep( self, whatPrintRep, whatPrintX, whatPrintY ):
    # Prints out the bit represention.

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

    def BuildLocalBitRep( self, localDimX, localDimY, centerX, centerY, ballX, ballY ):
    # Builds a bit-rep SDR of localDim dimensions centered around point with resolution.

        maxArraySize = self.resolutionX * self.resolutionY

        localBitRep = []

        scaleX = localDimX / self.resolutionX
        scaleY = localDimY / self.resolutionY

        if scaleX < 1 or scaleY < 1:
            sys.exit( "Resolution (X and Y) must be less than local dimensions (X and Y)." )

        # Right side border bits.
        if Within( self.screenWidth / 2, centerX - ( localDimX / 2 ), centerX + ( localDimX / 2 ), False ):
            for y in range( self.resolutionY ):
                bitToAdd = int( ( ( self.screenWidth / 2 ) - centerX + ( localDimX / 2 ) ) / scaleX ) + ( y * self.resolutionX )
                if bitToAdd >= maxArraySize or bitToAdd < 0:
                    sys.exit( "Right side border bit outside EncodingWidth size." )
                else:
                    localBitRep.append( bitToAdd )

        # Left side border bits.
        if Within( -self.screenWidth / 2, centerX - ( localDimX / 2 ), centerX + ( localDimX / 2 ), False ):
            for y in range( self.resolutionY ):
                bitToAdd = int( ( -( self.screenWidth / 2 ) - centerX + ( localDimX / 2 ) ) / scaleX ) + ( y * self.resolutionX )
                if bitToAdd >= maxArraySize or bitToAdd < 0:
                    sys.exit( "Left side border bit outside EncodingWidth size." )
                else:
                    localBitRep.append( bitToAdd )

        # Top side border bits.
        if Within( self.screenHeight / 2, centerY - ( localDimY / 2 ), centerY + ( localDimY / 2 ), False ):
            for x in range( self.resolutionX ):
                bitToAdd = x + ( int( ( (self.screenHeight / 2 ) - centerY + ( localDimY / 2 ) ) / scaleY ) * self.resolutionX )
                if bitToAdd >= maxArraySize or bitToAdd < 0:
                    sys.exit( "Top border bit outside EncodingWidth size:" )
                else:
                    localBitRep.append( bitToAdd )

        # Bottom side border bits.
        if Within( -self.screenHeight / 2, centerY - ( localDimY / 2 ), centerY + ( localDimY / 2 ), False ):
            for x in range( self.resolutionX ):
                bitToAdd = x + ( int( ( -(self.screenHeight / 2 ) - centerY + ( localDimY / 2 ) ) / scaleY ) * self.resolutionX )
                if bitToAdd >= maxArraySize or bitToAdd < 0:
                    sys.exit( "Bottom border bit outside EncodingWidth size." )
                else:
                    localBitRep.append( bitToAdd )

        # Ball bits.
        if Within( ballX, centerX - ( localDimX / 2 ) - ( self.ballWidth * 10 ), centerX + ( localDimX / 2 ) + ( self.ballWidth * 10 ), False ) and Within( ballY, centerY - ( localDimY / 2 ) - ( self.ballHeight * 10 ), centerY + ( localDimY / 2 ) + ( self.ballHeight * 10 ), False ):
            for x in range( self.ballWidth * 20 ):
                for y in range( self.ballHeight * 20 ):
                    if Within( ballX - ( self.ballWidth * 10 ) + x, centerX - ( localDimX / 2 ), centerX + ( localDimX / 2 ), False ) and Within( ballY - ( self.ballHeight * 10 ) + y, centerY - ( localDimY / 2 ), centerY + ( localDimY / 2 ), False ):
                        bitX = ballX - ( self.ballWidth * 10 ) + x - centerX + ( localDimX / 2 )
                        bitY = ballY - ( self.ballHeight * 10 ) + y - centerY + ( localDimY / 2 )
                        bitToAdd = int( bitX / scaleX ) + ( int( bitY / scaleY ) * self.resolutionX )
                        if bitToAdd >= maxArraySize or bitToAdd < 0:
                            sys.exit( "Ball bit outside EncodingWidth size." )
                        else:
                            localBitRep.append( bitToAdd )

#        self.PrintBitRep( localBitRep, self.resolutionX, self.resolutionY )

        bitRepSDR = SDR( maxArraySize )
        bitRepSDR.sparse = numpy.unique( localBitRep )
        return bitRepSDR

    def EncodeSenseData ( self, ballX, ballY, centerX, centerY, localDimX, localDimY ):
    # Encodes sense data as an SDR and returns it.

        centerXBits = self.centerXEncoder.encode( centerX )
        centerYBits = self.centerYEncoder.encode( centerY )
        bitRepresentation = self.BuildLocalBitRep( localDimX, localDimY, centerX, centerY, ballX, ballY )

        encoding = SDR( self.encodingWidth ).concatenate( [ bitRepresentation, centerXBits, centerYBits ] )
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    def LearnTimeStep ( self, secondLast, last, present, doLearn ):
    # Learn the three time-step data, from second last to last to present, centered around last time-step.

        self.tp.reset()

        if Within( secondLast[ 0 ] - last[ 0 ], -self.localDimX, self.localDimX, True ):
            if Within( secondLast[ 1 ] - last[ 1 ], -self.localDimY, self.localDimY, True ):
                secondLastSDR = self.EncodeSenseData( secondLast[ 0 ], secondLast[ 1 ], last[ 0 ], last[ 1 ], self.localDimX, self.localDimY )

                # Feed x and y position into classifier to learn.
                # Classifier can only take positive input, so need to transform ball origin.
                if doLearn:
                    self.xPosition.learn( pattern = secondLastSDR, classification = secondLast[ 0 ] - last[ 0 ] + int( self.localDimX / 2 ) )
                    self.yPosition.learn( pattern = secondLastSDR, classification = secondLast[ 1 ] - last[ 1 ] + int( self.localDimY / 2 ) )

                # Reset tp and feed SDR into tp.
                self.tp.compute( secondLastSDR, learn = doLearn )
                self.tp.activateDendrites( learn = doLearn )

        # Generate SDR for last sense data by feeding sense data into SP with learning.
        lastSDR = self.EncodeSenseData( 0, 0, last[ 0 ], last[ 1 ], self.localDimX, self.localDimY )

        if doLearn:
            self.xPosition.learn( pattern = lastSDR, classification = int( self.localDimX / 2 ) )
            self.yPosition.learn( pattern = lastSDR, classification = int( self.localDimY / 2 ) )

        self.tp.compute( lastSDR, learn = doLearn )
        self.tp.activateDendrites( learn = doLearn )

        if Within( present[ 0 ] - last[ 0 ], -self.localDimX, self.localDimX, True ):
            if Within( present[ 1 ] - last[ 1 ], -self.localDimY, self.localDimY, True ):
                # Generate SDR for last sense data by feeding sense data into SP with learning.
                senseSDR = self.EncodeSenseData( present[ 0 ], present[ 1 ], last[ 0 ], last[ 1 ], self.localDimX, self.localDimY )

                if doLearn:
                    self.xPosition.learn( pattern = senseSDR, classification = present[ 0 ] - last[ 0 ] + int( self.localDimX / 2 ) )
                    self.yPosition.learn( pattern = senseSDR, classification = present[ 1 ] - last[ 1 ] + int( self.localDimY / 2 ) )

                self.tp.compute( senseSDR, learn = doLearn )
                self.tp.activateDendrites( learn = doLearn )

    def PredictTimeStep ( self, secondLast, last, doLearn ):
    # Train time-step data, from secondlast to last, centered around last, and then predict next position and return.

        self.tp.reset()

        if Within( secondLast[ 0 ] - last[ 0 ], -self.localDimX, self.localDimX, True ):
            if Within( secondLast[ 1 ] - last[ 1 ], -self.localDimY, self.localDimY, True ):
                secondLastSDR = self.EncodeSenseData( secondLast[ 0 ], secondLast[ 1 ], last[ 0 ], last[ 1 ], self.localDimX, self.localDimY )
                self.tp.compute( secondLastSDR, learn = doLearn )
                self.tp.activateDendrites( learn = doLearn )

        lastSDR = self.EncodeSenseData( 0, 0, last[ 0 ], last[ 1 ], self.localDimX, self.localDimY )
        self.tp.compute( lastSDR, learn = doLearn )
        self.tp.activateDendrites( learn = doLearn )
        predictCellsTP = self.tp.getPredictiveCells()

        # Get predicted location for next time step.
        stepSenseSDR = SDR( self.sp.getColumnDimensions() )
        stepSenseSDR.sparse = numpy.unique( [ self.tp.columnForCell( cell ) for cell in predictCellsTP.sparse ] )
        positionX = numpy.argmax( self.xPosition.infer( pattern = stepSenseSDR ) ) - int( self.localDimX / 2 )
        positionY = numpy.argmax( self.yPosition.infer( pattern = stepSenseSDR ) ) - int( self.localDimY / 2 )

        return [ last[ 0 ] + positionX, last[ 1 ] + positionY, last[ 2 ] ]

    def Brain ( self, ballX, ballY, paddleY ):
    # Agents brain center.

        # Save present ball coordinates in memory.
        self.memBuffer.append( [ ballX, ballY, paddleY ] )
        while len( self.memBuffer ) > self.maxMemoryDist:
            self.memBuffer.pop( 0 )

        self.LearnTimeStep( self.memBuffer[ -3 ], self.memBuffer[ -2 ], self.memBuffer[ -1 ], True )

        # Store last and present locations.
        self.predPositions.clear()
        self.predPositions.append( self.memBuffer[ -2 ] )
        self.predPositions.append( self.memBuffer[ -1 ] )

        # Predict next 10 time step locations and store them.
        for step in range( self.maxMemoryDist ):
            nextPosition = self.PredictTimeStep( self.predPositions[ -2 ], self.predPositions[ -1 ], False )
            if Within( nextPosition[ 0 ], -self.screenWidth / 2, self.screenWidth / 2, True ):
                if Within( nextPosition[ 1 ], -self.screenHeight / 2, self.screenHeight / 2, True ):
                    self.predPositions.append( nextPosition )
                else:
                    break
            else:
                break
