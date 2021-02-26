import numpy
import sys
import random

from htm.bindings.sdr import SDR, Metrics
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.algorithms import Classifier
import htm.bindings.encoders
ScalarEncoder = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters

class BallAgent:

    resolutionX = 20                # Should be an even number
    resolutionY = 20
    localDimX   = 100
    localDimY   = 100

    def __init__( self, name, screenHeight, screenWidth, ballHeight, ballWidth, paddleHeight, paddleWidth ):

        self.ID = name

        self.screenHeight = screenHeight
        self.screenWidth  = screenWidth
        self.ballHeight   = ballHeight
        self.ballWidth    = ballWidth
        self.paddleHeight = paddleHeight
        self.paddleWidth  = paddleWidth

        centerVelXEncodeParams  = ScalarEncoderParameters()
        centerVelYEncodeParams  = ScalarEncoderParameters()

        centerVelXEncodeParams.activeBits = 10
        centerVelXEncodeParams.radius     = 5
        centerVelXEncodeParams.clipInput  = False
        centerVelXEncodeParams.minimum    = -30
        centerVelXEncodeParams.maximum    = 30
        centerVelXEncodeParams.periodic   = False

        centerVelYEncodeParams.activeBits = 10
        centerVelYEncodeParams.radius     = 5
        centerVelYEncodeParams.clipInput  = False
        centerVelYEncodeParams.minimum    = -30
        centerVelYEncodeParams.maximum    = 30
        centerVelYEncodeParams.periodic   = False

        self.centerVelXEncoder = ScalarEncoder( centerVelXEncodeParams )
        self.centerVelYEncoder = ScalarEncoder( centerVelYEncodeParams )

        self.aspEncodingWidth = ( ( self.resolutionX * self.resolutionY ) + self.centerVelXEncoder.size + self.centerVelYEncoder.size )

        self.asp = SpatialPooler(
            inputDimensions            = ( self.aspEncodingWidth, ),
            columnDimensions           = ( 2048, ),
            potentialPct               = 0.85,
            potentialRadius            = self.aspEncodingWidth,
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

        self.atp = TemporalMemory(
            columnDimensions          = ( 2048, ),
            cellsPerColumn            = 6,
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

        self.centerXVelClassifier  = Classifier( alpha = 1 )
        self.centerYVelClassifier  = Classifier( alpha = 1 )

        self.centerX    = 0
        self.centerY    = 0
        self.centerVelX = 0
        self.centerVelY = 0

        self.predPositions = []
        self.lastSenseSDR  = SDR( self.asp.getColumnDimensions() )

    def Within ( self, value, minimum, maximum, equality ):
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

    def EncodeSenseData ( self, bitRepresentation, centerVelX, centerVelY ):
    # Encodes sense data as an SDR and returns it.

        centerVelXBits = self.centerVelXEncoder.encode( centerVelX )
        centerVelYBits = self.centerVelYEncoder.encode( centerVelY )

        encoding = SDR( self.aspEncodingWidth ).concatenate( [ bitRepresentation, centerVelXBits, centerVelYBits ] )
        senseSDR = SDR( self.asp.getColumnDimensions() )
        self.asp.compute( encoding, True, senseSDR )

        return senseSDR

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
        if self.Within( self.screenWidth / 2, centerX - ( localDimX / 2 ), centerX + ( localDimX / 2 ), False ):
            for y in range( self.resolutionY ):
                bitToAdd = int( ( ( self.screenWidth / 2 ) - centerX + ( localDimX / 2 ) ) / scaleX ) + ( y * self.resolutionX )
                if bitToAdd >= maxArraySize or bitToAdd < 0:
                    sys.exit( "Right side border bit outside aspEncodingWidth size." )
                else:
                    localBitRep.append( bitToAdd )

        # Left side border bits.
        if self.Within( -self.screenWidth / 2, centerX - ( localDimX / 2 ), centerX + ( localDimX / 2 ), False ):
            for y in range( self.resolutionY ):
                bitToAdd = int( ( -( self.screenWidth / 2 ) - centerX + ( localDimX / 2 ) ) / scaleX ) + ( y * self.resolutionX )
                if bitToAdd >= maxArraySize or bitToAdd < 0:
                    sys.exit( "Left side border bit outside aspEncodingWidth size." )
                else:
                    localBitRep.append( bitToAdd )

        # Top side border bits.
        if self.Within( self.screenHeight / 2, centerY - ( localDimY / 2 ), centerY + ( localDimY / 2 ), False ):
            for x in range( self.resolutionX ):
                bitToAdd = x + ( int( ( (self.screenHeight / 2 ) - centerY + ( localDimY / 2 ) ) / scaleY ) * self.resolutionX )
                if bitToAdd >= maxArraySize or bitToAdd < 0:
                    sys.exit( "Top border bit outside aspEncodingWidth size:" )
                else:
                    localBitRep.append( bitToAdd )

        # Bottom side border bits.
        if self.Within( -self.screenHeight / 2, centerY - ( localDimY / 2 ), centerY + ( localDimY / 2 ), False ):
            for x in range( self.resolutionX ):
                bitToAdd = x + ( int( ( -(self.screenHeight / 2 ) - centerY + ( localDimY / 2 ) ) / scaleY ) * self.resolutionX )
                if bitToAdd >= maxArraySize or bitToAdd < 0:
                    sys.exit( "Bottom border bit outside aspEncodingWidth size." )
                else:
                    localBitRep.append( bitToAdd )

        # Ball bits.
        if self.Within( ballX, centerX - ( localDimX / 2 ) - ( self.ballWidth * 10 ), centerX + ( localDimX / 2 ) + ( self.ballWidth * 10 ), False ) and self.Within( ballY, centerY - ( localDimY / 2 ) - ( self.ballHeight * 10 ), centerY + ( localDimY / 2 ) + ( self.ballHeight * 10 ), False ):
            for x in range( self.ballWidth * 20 ):
                for y in range( self.ballHeight * 20 ):
                    if self.Within( ballX - ( self.ballWidth * 10 ) + x, centerX - ( localDimX / 2 ), centerX + ( localDimX / 2 ), False ) and self.Within( ballY - ( self.ballHeight * 10 ) + y, centerY - ( localDimY / 2 ), centerY + ( localDimY / 2 ), False ):
                        bitX = ballX - ( self.ballWidth * 10 ) + x - centerX + ( localDimX / 2 )
                        bitY = ballY - ( self.ballHeight * 10 ) + y - centerY + ( localDimY / 2 )
                        bitToAdd = int( bitX / scaleX ) + ( int( bitY / scaleY ) * self.resolutionX )
                        if bitToAdd >= maxArraySize or bitToAdd < 0:
                            sys.exit( "Ball bit outside aspEncodingWidth size." )
                        else:
                            localBitRep.append( bitToAdd )

#        self.PrintBitRep( localBitRep, self.resolutionX, self.resolutionY )

        bitRepSDR = SDR( maxArraySize )
        bitRepSDR.sparse = numpy.unique( localBitRep )
        return bitRepSDR

    def PredictTimeStep ( self ):
    # Use subsequence ball positions shift and btp to predict next number of ball positions.

        self.predPositions.clear()

        self.atp.reset()
        self.atp.compute( self.lastSenseSDR, learn = True )
        self.atp.activateDendrites( learn = True )

        tempCenterX    = self.centerX
        tempCenterY    = self.centerY
        tempCenterVelX = self.centerVelX
        tempCenterVelY = self.centerVelY

        tempSenseSDR = SDR( self.asp.getColumnDimensions() )

        # Predict next number of ball positions.
        for predSteps in range( 10 ):
            # Generate tempSenseSDR
            tempSenseSDR = self.EncodeSenseData( self.BuildLocalBitRep( self.localDimX, self.localDimY, tempCenterX, tempCenterY, tempCenterX, tempCenterY ),
                tempCenterVelX, tempCenterVelY )

            # Generate predictive cells.
            self.atp.reset()
            self.atp.compute( tempSenseSDR, learn = False )
            self.atp.activateDendrites( learn = False )
            predictCellsTP = self.atp.getPredictiveCells()

            tempSenseSDR.sparse = numpy.unique( [ self.atp.columnForCell( cell ) for cell in predictCellsTP.sparse ] )

            # Infer the position shift from this subsequence, and apply it to our temp position.
            tempCenterVelX = numpy.argmax( self.centerXVelClassifier.infer( pattern = tempSenseSDR ) ) - 100
            tempCenterVelY = numpy.argmax( self.centerYVelClassifier.infer( pattern = tempSenseSDR ) ) - 100

            if tempCenterVelX < -30:
                tempCenterVelX = -30
            if tempCenterVelY < -30:
                tempCenterVelY = -30

            tempCenterX += tempCenterVelX
            if tempCenterX > self.screenWidth / 2:
                tempCenterX = self.screenWidth / 2
            elif tempCenterX < -self.screenWidth / 2:
                tempCenterX = -self.screenWidth / 2
            tempCenterY += tempCenterVelY
            if tempCenterY > self.screenHeight / 2:
                tempCenterY = self.screenHeight / 2
            elif tempCenterY < -self.screenHeight / 2:
                tempCenterY = -self.screenHeight / 2

            # Append this position as a prediction.
            self.predPositions.append( [ tempCenterX, tempCenterY ] )

    def Brain ( self, ballX, ballY, ballXSpeed, ballYSpeed ):
    # Agents brain center.

        self.centerX    = ballX
        self.centerY    = ballY
        self.centerVelX = ballXSpeed
        self.centerVelY = ballYSpeed

        # Feed in attention view to encode bit representation and produce SDR.
        senseSDR = self.EncodeSenseData( self.BuildLocalBitRep( self.localDimX, self.localDimY, self.centerX, self.centerY, ballX, ballY ),
            self.centerVelX, self.centerVelY )

        self.centerXVelClassifier.learn( pattern = senseSDR, classification = self.centerVelX + 100 )
        self.centerYVelClassifier.learn( pattern = senseSDR, classification = self.centerVelY + 100 )

        # Feed SDR into atp to learn as part of sequence.
        self.atp.reset()
        self.atp.compute( self.lastSenseSDR, learn = True )
        self.atp.activateDendrites( learn = True )
        self.atp.compute( senseSDR, learn = True )
        self.atp.activateDendrites( learn = True )

        self.lastSenseSDR = senseSDR

        # Generate predictions.
        self.PredictTimeStep()
