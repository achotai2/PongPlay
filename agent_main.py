import numpy
import random

from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
import htm.bindings.encoders
ScalarEncoder = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters
RDSE = htm.bindings.encoders.RDSE
RDSE_Parameters = htm.bindings.encoders.RDSE_Parameters
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.algorithms import Classifier

class Agent:

    testThreshold   = 30
    burstThreshold  = 5
    goalThreshold   = 1.0
    motorDimensions = 3
    synapseInc      = 0.05
    synapseDec      = 0.01

    def __init__( self, name, screenHeight, screenWidth ):
        self.ID = name

        self.senseBuffer = []       # Memory of all [ senseSDR, winningMotor ] experienced this sequence.

        # Set up encoder parameters
        paddleEncodeParams    = ScalarEncoderParameters()
        ballXEncodeParams     = ScalarEncoderParameters()
        ballYEncodeParams     = ScalarEncoderParameters()
        ballVelEncodeParams   = RDSE_Parameters()
        manInEncodeParams     = ScalarEncoderParameters()

        paddleEncodeParams.activeBits = 99
        paddleEncodeParams.radius     = 200
        paddleEncodeParams.clipInput  = False
        paddleEncodeParams.minimum    = -screenHeight / 2
        paddleEncodeParams.maximum    = screenHeight / 2
        paddleEncodeParams.periodic   = False

        ballXEncodeParams.activeBits = 21
        ballXEncodeParams.radius     = 40
        ballXEncodeParams.clipInput  = False
        ballXEncodeParams.minimum    = -screenWidth / 2
        ballXEncodeParams.maximum    = screenWidth / 2
        ballXEncodeParams.periodic   = False

        ballYEncodeParams.activeBits = 21
        ballYEncodeParams.radius     = 40
        ballYEncodeParams.clipInput  = False
        ballYEncodeParams.minimum    = -screenHeight
        ballYEncodeParams.maximum    = screenHeight
        ballYEncodeParams.periodic   = False

        ballVelEncodeParams.size       = 400
        ballVelEncodeParams.activeBits = 25
        ballVelEncodeParams.resolution = 0.1

        manInEncodeParams.activeBits = 10
        manInEncodeParams.radius     = 1
        manInEncodeParams.clipInput  = False
        manInEncodeParams.minimum    = -1
        manInEncodeParams.maximum    = 2
        manInEncodeParams.periodic   = False

        # Set up encoders
        self.paddleEncoder   = ScalarEncoder( paddleEncodeParams )
        self.ballEncoderX    = ScalarEncoder( ballXEncodeParams )
        self.ballEncoderY    = ScalarEncoder( ballYEncodeParams )
        self.ballEncoderVelX = RDSE( ballVelEncodeParams )
        self.ballEncoderVelY = RDSE( ballVelEncodeParams )
        self.manInEncoder    = ScalarEncoder( manInEncodeParams )

        self.encodingWidth = ( self.paddleEncoder.size + self.ballEncoderX.size + self.ballEncoderY.size +
            self.ballEncoderVelX.size + self.ballEncoderVelY.size + self.manInEncoder.size )

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

        self.motorSynapse = numpy.random.uniform(
            low = 0.0, high = 1.0,
            size = ( self.tp.getColumnDimensions()[ 0 ] * self.tp.getCellsPerColumn() * self.motorDimensions, )
        )

        self.manualInput = -1

        self.numEvents = 0
        self.percentSuccess = 0

    def Clear ( self ):
    # Clear sense buffer.
        self.senseBuffer.clear()
        self.tp.reset()


    def SendSuggest ( self, input ):
    # Sets the movement suggestion from user. manualInput = -1 means no suggestion.
        self.manualInput = input
        self.tp.reset()

    def EncodeSenseData ( self, yPos, ballX, ballY, ballXSpeed, ballYSpeed ):
    # Encodes sense data as an SDR and returns it.

        # Now we call the encoders to create bit representations for each value, and encode them.
        paddleBits   = self.paddleEncoder.encode( yPos )
        ballBitsX    = self.ballEncoderX.encode( ballX )
        ballBitsY    = self.ballEncoderY.encode( ballY )
        ballBitsVelX = self.ballEncoderVelX.encode( ballXSpeed )
        ballBitsVelY = self.ballEncoderVelY.encode( ballYSpeed )
        manInBits    = self.manInEncoder.encode( self.manualInput )

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( self.encodingWidth ).concatenate( [ paddleBits, ballBitsX, ballBitsY, ballBitsVelX, ballBitsVelY, manInBits ] )
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    # NOT USED CURRENTLY
    def Overlap ( self, SDR1, SDR2 ):
    # Computes overlap score between two passed SDRs.

        overlap = 0

        for cell1 in SDR1.sparse:
            if cell1 in SDR2.sparse:
                overlap += 1

        return overlap

    # NOT USED CURRENTLY
    def GreatestOverlap ( self, testSDR, listSDR, threshold ):
    # Finds SDR in listSDR with greatest overlap with testSDR and returns it, and its index in the list.
    # If none are found above threshold or if list is empty it returns an empty SDR of length testSDR, with index -1.

        greatest = [ SDR( testSDR.size ), -1 ]

        if len(listSDR) > 0:
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

    def UpdateScore ( self, successEvent ):
    # Updates agents event and success percentage.

        numSuccess = self.numEvents * self.percentSuccess / 100
        self.numEvents += 1
        if successEvent:
            numSuccess += 1
        self.percentSuccess = 100 * numSuccess / self.numEvents

    def DetermineBurstPercent ( self ):
    # Calculates percentage of active cells that are currently bursting.

        activeCellsTP = self.tp.getActiveCells()

        # Get columns of all active cells.
        activeColumnsTP = []
        for cCell in activeCellsTP.sparse:
            activeColumnsTP.append( self.tp.columnForCell( cCell ) )

        # Get count of active cells in each active column.
        colUnique, colCount = numpy.unique( activeColumnsTP, return_counts = True )

        # Compute percentage of columns that are bursting.
        bursting = 0
        for c in colCount:
            if c > 1:
                bursting += 1
        burstPercent = int( 100 * bursting / len( colCount ) )

        return burstPercent

    def Hippocampus ( self, feeling ):
    # Learns sequence back sequenceLength-time steps in memory, then stores sequence along with feeling.

        if feeling > 1.0 or feeling < -1.0:
            sys.exit( "Feeling states should be in the range [-1.0, 1.0]" )

        for buffEntry in self.senseBuffer:
            if buffEntry[ 2 ]:
                for cell in buffEntry[ 1 ].sparse:
                    for i in range( self.motorDimensions ):
                        if i == buffEntry[ 0 ]:
                            self.motorSynapse[ ( cell * self.motorDimensions ) + i ] += self.synapseInc * feeling
                            if self.motorSynapse[ ( cell * self.motorDimensions ) + i ] > 1.0:
                                self.motorSynapse[ ( cell * self.motorDimensions ) + i ] = 1.0
                        else:
                            self.motorSynapse[ ( cell * self.motorDimensions ) + i ] -= self.synapseDec * feeling
                            if self.motorSynapse[ ( cell * self.motorDimensions ) + i ] < 0.0:
                                self.motorSynapse[ ( cell * self.motorDimensions ) + i ] = 0.0

    def Brain ( self, yPos, ballX, ballY, ballXSpeed, ballYSpeed ):
    # Agents brain center.

        # Generate SDR for sense data by feeding sense data into SP with learning.
        senseSDR = self.EncodeSenseData( yPos, ballX, ballY, ballXSpeed, ballYSpeed )

        # Feed present senseSDR into tp and generate active cells.
        self.tp.compute( senseSDR, learn = True )
        self.tp.activateDendrites( learn = True )
        winnerCellsTP = self.tp.getWinnerCells()

        motorScore = [ 0.0, 0.0, 0.0 ]         # Keeps track of each motor output weighted score [ UP, STILL, DOWN ]
        # Use active cells to determine motor action by feeding through motorSynapse.
        for cell in winnerCellsTP.sparse:
            for i in range( self.motorDimensions ):
                motorScore[i] += self.motorSynapse[ ( cell * self.motorDimensions ) + i ]

        # Use largest motorScore to choose winningMotor action.
        largest = []
        for i, v in enumerate( motorScore ):
            if sorted( motorScore, reverse=True )[ 0 ] == v:
                largest.append( i )                                         # 0 = UP, 1 = STILL, 2 = DOWN
        winningMotor = random.choice( largest )

        # Determine if cells are bursting above some percent threshold or not.
        if self.DetermineBurstPercent() <= self.burstThreshold:
            isPredicted = True
        else:
            isPredicted = False

        # Add senseSDR and winningMotor to buffer.
        self.senseBuffer.insert( 0, [ winningMotor, winnerCellsTP, isPredicted ] )

#        if self.manualInput != -1:
#            # If motor suggestion, manualInput, equals winningMotor then send a small reward.
#            if winningMotor == self.manualInput:
#                self.Hippocampus( 0.1 )
#            # If not send a small punishment.
#            else:
#                self.Hippocampus( -0.1 )

        # Return winning motor function.
        return winningMotor
