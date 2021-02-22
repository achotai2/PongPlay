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

    motorDimensions = 3
    subSeqThresh  = 18


    resolutionX = 50                # Should be an even number
    resolutionY = 50
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

        self.aspEncodingWidth = ( ( self.resolutionX * self.resolutionY ) + self.centerXEncoder.size
            + self.centerYEncoder.size + self.centerVelXEncoder.size + self.centerVelYEncoder.size )

        self.asp = SpatialPooler(
            inputDimensions            = ( self.aspEncodingWidth, ),
            columnDimensions           = ( 1024, ),
            potentialPct               = 0.85,
            potentialRadius            = self.aspEncodingWidth,
            globalInhibition           = True,
            localAreaDensity           = 0,
            numActiveColumnsPerInhArea = 20,
            synPermInactiveDec         = 0.005,
            synPermActiveInc           = 0.04,
            synPermConnected           = 0.1,
            boostStrength              = 3.0,
            seed                       = -1,
            wrapAround                 = False
        )

        self.atp = TemporalMemory(
            columnDimensions          = ( 1024, ),
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

        centerXEncodeParams     = ScalarEncoderParameters()
        centerYEncodeParams     = ScalarEncoderParameters()
        centerVelXEncodeParams  = ScalarEncoderParameters()
        centerVelYEncodeParams  = ScalarEncoderParameters()

        centerXEncodeParams.activeBits = 5
        centerXEncodeParams.radius     = 10
        centerXEncodeParams.clipInput  = False
        centerXEncodeParams.minimum    = -screenWidth / 2
        centerXEncodeParams.maximum    = screenWidth / 2
        centerXEncodeParams.periodic   = False

        centerYEncodeParams.activeBits = 5
        centerYEncodeParams.radius     = 10
        centerYEncodeParams.clipInput  = False
        centerYEncodeParams.minimum    = -screenHeight / 2
        centerYEncodeParams.maximum    = screenHeight / 2
        centerYEncodeParams.periodic   = False

        centerVelXEncodeParams.activeBits = 5
        centerVelXEncodeParams.radius     = 5
        centerVelXEncodeParams.clipInput  = False
        centerVelXEncodeParams.minimum    = -30
        centerVelXEncodeParams.maximum    = 30
        centerVelXEncodeParams.periodic   = False

        centerVelYEncodeParams.activeBits = 5
        centerVelYEncodeParams.radius     = 5
        centerVelYEncodeParams.clipInput  = False
        centerVelYEncodeParams.minimum    = -30
        centerVelYEncodeParams.maximum    = 30
        centerVelYEncodeParams.periodic   = False

        self.centerXEncoder    = ScalarEncoder( centerXEncodeParams )
        self.centerYEncoder    = ScalarEncoder( centerYEncodeParams )
        self.centerVelXEncoder = ScalarEncoder( centerVelXEncodeParams )
        self.centerVelYEncoder = ScalarEncoder( centerVelYEncodeParams )

#        self.bspEncodingWidth = ( ( self.atp.getColumnDimensions()[ 0 ] * self.atp.getCellsPerColumn() )
#            + self.centerXEncoder.size + self.centerYEncoder.size + self.centerVelXEncoder.size
#            + self.centerVelYEncoder.size )

#        self.bsp = SpatialPooler(
#            inputDimensions            = ( self.bspEncodingWidth, ),
#            columnDimensions           = ( 1024, ),
#            potentialPct               = 0.85,
#            potentialRadius            = self.bspEncodingWidth,
#            globalInhibition           = True,
#            localAreaDensity           = 0,
#            numActiveColumnsPerInhArea = 20,
#            synPermInactiveDec         = 0.005,
#            synPermActiveInc           = 0.04,
#            synPermConnected           = 0.1,
#            boostStrength              = 3.0,
#            seed                       = -1,
#            wrapAround                 = False
#        )

#        self.btp = TemporalMemory(
#            columnDimensions          = ( 1024, ),
#            cellsPerColumn            = 8,
#            activationThreshold       = 16,
#            initialPermanence         = 0.21,
#            connectedPermanence       = 0.1,
#            minThreshold              = 12,
#            maxNewSynapseCount        = 20,
#            permanenceIncrement       = 0.1,
#            permanenceDecrement       = 0.1,
#            predictedSegmentDecrement = 0.0,
#            maxSegmentsPerCell        = 128,
#            maxSynapsesPerSegment     = 32,
#            seed                      = 42
#        )

        self.centerXPosition  = Classifier( alpha = 1 )
        self.centerYPosition  = Classifier( alpha = 1 )
        self.centerXVelocity  = Classifier( alpha = 1 )
        self.centerYVelocity  = Classifier( alpha = 1 )
#        self.subSequenceClass = Classifier( alpha = 1 )

        self.centerX    = 0
        self.centerY    = 0
        self.centerVelX = 0
        self.centerVelY = 0

        self.predPositions = []
#        self.subSequences  = []
#        self.lastActiveATP = SDR( self.atp.getColumnDimensions()[ 0 ] * self.atp.getCellsPerColumn() )
        self.lastBroadSDR  = SDR( self.bsp.getColumnDimensions() )

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

    def Overlap ( self, SDR1, SDR2 ):
    # Computes overlap score between two passed SDRs.

        overlap = 0

        for cell1 in SDR1.sparse:
            if cell1 in SDR2.sparse:
                overlap += 1

        return overlap

    def GreatestOverlap ( self, testSDR, listSDR, threshold ):
    # Finds SDR in listSDR with greatest overlap with testSDR and returns it, and its index in the list.
    # If none are found above threshold or if list is empty it returns an empty SDR of length testSDR, with index -1.

        greatest = [ SDR( testSDR.size ), -1 ]

        if len( listSDR ) > 0:
            # The first element of listSDR should always be a union of all the other SDRs in list,
            # so a check can be performed first.
            if self.Overlap( testSDR, listSDR[0] ) >= threshold:
                aboveThreshold = []
                for idx, checkSDR in enumerate( listSDR ):
                    if idx != 0:
                        thisOverlap = self.Overlap( testSDR, checkSDR )
                        if thisOverlap >= threshold:
                            aboveThreshold.append( [ thisOverlap, [checkSDR, idx] ] )
                if len( aboveThreshold ) > 0:
                    greatest = sorted( aboveThreshold, key = lambda tup: tup[0], reverse = True )[ 0 ][ 1 ]

        return greatest

    def EncodeSenseData ( self, bitRepresentation, centerX, centerY, centerVelX, centerVelY ):
    # Encodes sense data as an SDR and returns it.

        centerXBits       = self.centerXEncoder.encode( centerX )
        centerYBits       = self.centerYEncoder.encode( centerY )
        centerVelXBits    = self.centerXEncoder.encode( centerVelX )
        centerVelYBits    = self.centerYEncoder.encode( centerVelY )

        encoding = SDR( self.aspEncodingWidth ).concatenate( [ bitRepresentation, centerXBits, centerYBits, centerVelXBits, centerVelYBits ] )
        senseSDR = SDR( self.asp.getColumnDimensions() )
        self.asp.compute( encoding, True, senseSDR )

        return senseSDR

    def EncodeBroadData( self, attentionSDR, centerX, centerY ):
    # Encodes the data for the broad system.

        # Now we call the encoders to create bit representations for each value, and encode them.
        centerXBits       = self.centerXEncoder.encode( centerX )
        centerYBits       = self.centerYEncoder.encode( centerY )
        flatAttentionBits = SDR( self.atp.getColumnDimensions()[ 0 ] * self.atp.getCellsPerColumn() )
        flatAttentionBits.sparse = attentionSDR.sparse

        encoding = SDR( self.bspEncodingWidth ).concatenate( [ centerXBits, centerYBits, flatAttentionBits ] )
        senseSDR = SDR( self.bsp.getColumnDimensions() )
        self.bsp.compute( encoding, True, senseSDR )

        return senseSDR

    def PrintBitRep( self, whatPrintRep, whatPrintX, whatPrintY )
    # Prints out the bit represention.

        for y in range( whatPrintRep ):
            for x in range( whatPrintRep ):
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

        localBitRep = []

        scaleX = localDimX / self.resolutionX
        scaleY = localDimY / self.resolutionY

        if scaleX < 1 or scaleY < 1:
            sys.exit( "Resolution (X and Y) must be less than local dimensions (X and Y)." )

        # Right side border bits.
        if self.Within( self.screenWidth / 2, centerX - ( localDimX / 2 ), centerX + ( localDimX / 2 ), False ):
            for y in range( self.resolutionY ):
                bitToAdd = int( ( ( self.screenWidth / 2 ) - centerX + ( localDimX / 2 ) ) / scaleX ) + ( y * self.resolutionX )
                if bitToAdd >= self.aspEncodingWidth or bitToAdd < 0:
                    sys.exit( "Right side border bit outside aspEncodingWidth size." )
                else:
                    localBitRep.append( bitToAdd )

        # Left side border bits.
        if self.Within( -self.screenWidth / 2, centerX - ( localDimX / 2 ), centerX + ( localDimX / 2 ), False ):
            for y in range( self.resolutionY ):
                bitToAdd = int( ( -( self.screenWidth / 2 ) - centerX + ( localDimX / 2 ) ) / scaleX ) + ( y * self.resolutionX )
                if bitToAdd >= self.aspEncodingWidth or bitToAdd < 0:
                    sys.exit( "Left side border bit outside aspEncodingWidth size." )
                else:
                    localBitRep.append( bitToAdd )

        # Top side border bits.
        if self.Within( self.screenHeight / 2, centerY - ( localDimY / 2 ), centerY + ( localDimY / 2 ), False ):
            for x in range( self.resolutionX ):
                bitToAdd = x + ( int( ( (self.screenHeight / 2 ) - centerY + ( localDimY / 2 ) ) / scaleY ) * self.resolutionX )
                if bitToAdd >= self.aspEncodingWidth or bitToAdd < 0:
                    sys.exit( "Top border bit outside aspEncodingWidth size:" )
                else:
                    localBitRep.append( bitToAdd )

        # Bottom side border bits.
        if self.Within( -self.screenHeight / 2, centerY - ( localDimY / 2 ), centerY + ( localDimY / 2 ), False ):
            for x in range( self.resolutionX ):
                bitToAdd = x + ( int( ( -(self.screenHeight / 2 ) - centerY + ( localDimY / 2 ) ) / scaleY ) * self.resolutionX )
                if bitToAdd >= self.aspEncodingWidth or bitToAdd < 0:
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
                        if bitToAdd >= self.aspEncodingWidth or bitToAdd < 0:
                            sys.exit( "Ball bit outside aspEncodingWidth size." )
                        else:
                            localBitRep.append( bitToAdd )

#        self.PrintBitRep( localBitRep, self.resolutionX, self.resolutionY )

        bitRepSDR = SDR( self.aspEncodingWidth )
        bitRepSDR.sparse = numpy.unique( localBitRep )
        return bitRepSDR

    def PredictTimeStep ( self, currentBroadSDR ):
    # Use subsequence ball positions shift and btp to predict next number of ball positions.

        self.predPositions.clear()

        tempBroadSDR = SDR( currentBroadSDR.size )
        tempBroadSDR.sparse = currentBroadSDR.sparse

        tempCenterX = self.centerX
        tempCenterY = self.centerY

        # Predict next number of ball positions.
        for predSteps in range( 10 ):
            # Infer what the subsequence cell activation is from tempBroadSDR.
            subSequenceActive = numpy.argmax( self.subSequenceClass.infer( pattern = tempBroadSDR ) )

            # Infer the position shift from this subsequence, and apply it to our temp position.
            shiftX = numpy.argmax( self.xPosition.infer( pattern = self.subSequences[ subSequenceActive ] ) ) - 100
            shiftY = numpy.argmax( self.yPosition.infer( pattern = self.subSequences[ subSequenceActive ] ) ) - 100
            tempCenterX += shiftX
            if tempCenterX > self.screenWidth / 2:
                tempCenterX = self.screenWidth / 2
            elif tempCenterX < -self.screenWidth / 2:
                tempCenterX = -self.screenWidth / 2
            tempCenterY += shiftY
            if tempCenterY > self.screenHeight / 2:
                tempCenterY = self.screenHeight / 2
            elif tempCenterY < -self.screenHeight / 2:
                tempCenterY = -self.screenHeight / 2

            # Append this position as a prediction.
            self.predPositions.append( [ tempCenterX, tempCenterY ] )

            # Plug in our current data to get tempBroadSDR for next predicted timestep.
            tempBroadSDR = self.EncodeBroadData( self.subSequences[ subSequenceActive ], tempCenterX, tempCenterY )

            self.btp.reset()
            self.btp.compute( tempBroadSDR, learn = False )
            self.btp.activateDendrites( learn = False )
            predictCellsTP = self.btp.getPredictiveCells()

            tempBroadSDR.sparse = numpy.unique( [ self.btp.columnForCell( cell ) for cell in predictCellsTP.sparse ] )

    def Brain ( self, ballX, ballY, ballXSpeed, ballYSpeed ):
    # Agents brain center.

        self.centerX    = ballX
        self.centerY    = ballY
        self.centerVelX = ballXSpeed
        self.centerVelY = ballYSpeed

        # If ball has moved out of attention view then shift attention view to ball and reset local sequence.
        if not self.Within( ballX, self.centerX - ( self.localDimX / 2 ), self.centerX + ( self.localDimX / 2 ), True ) or not self.Within( ballY, self.centerY - ( self.localDimY / 2 ), self.centerY + ( self.localDimY / 2 ), True ):
            # Check if lastActiveATP is already stored in memory.
            isSubSequence = self.GreatestOverlap( self.lastActiveATP, self.subSequences, self.subSeqThresh )
            if isSubSequence[ 1 ] == -1:
                # If it isn't then add it.
                if len( self.subSequences ) == 0:
                    self.subSequences.append( self.lastActiveATP )
                self.subSequences.append( self.lastActiveATP )
                self.subSequences[ 0 ].sparse = numpy.union1d( self.lastActiveATP.sparse, self.subSequences[ 0 ].sparse )

                isSubSequence[ 1 ] = len( self.subSequences ) - 1

            # Feed last active cells from atp into btp.
            broadSenseSDR = self.EncodeBroadData( self.lastActiveATP, self.centerX, self.centerY )

            # Train the subSequenceClass classifier on broadSenseSDR and the index of subsequence in storage.
            self.subSequenceClass.learn( pattern = broadSenseSDR, classification = isSubSequence[ 1 ] )

            # Learn the active cell encoding as a ball shift (include a shift since always must be positive).
            self.xPosition.learn( pattern = self.lastActiveATP, classification = ballX - self.centerX + 100 )
            self.yPosition.learn( pattern = self.lastActiveATP, classification = ballY - self.centerY + 100 )

            # Feed broadSenseSDR into btp and learn it as a two-step sequence.
            self.btp.reset()
            self.btp.compute( broadSenseSDR, learn = True )
            self.btp.activateDendrites( learn = True )
            self.btp.compute( broadSenseSDR, learn = True )
            self.btp.activateDendrites( learn = True )

            # Set broadSenseSDR as lastBroadSDR for next time.
            self.lastBroadSDR = broadSenseSDR

            # Run prediction on current broadSenseSDR.
            self.PredictTimeStep( broadSenseSDR )

            # Move center onto ball, and rest atp.
            self.centerX = ballX
            self.centerY = ballY
            self.atp.reset()

        # Feed in attention view to encode bit representation and produce SDR.
        senseSDR = self.EncodeSenseData( self.BuildLocalBitRep( self.localDimX, self.localDimY, self.centerX, self.centerY, ballX, ballY ) )

        # Feed SDR into atp to learn as part of sequence.
        self.atp.compute( senseSDR, learn = True )
        self.atp.activateDendrites( learn = True )
        activeCellsTP = self.atp.getActiveCells()
        self.lastActiveATP = activeCellsTP

        print("Ball X:", ballX, "Ball Y:", ballY, "Speed X:", ballXSpeed, "Speed Y:", ballYSpeed )
        print(self.lastActiveATP)
        print("---------------------------------------------------------------")
