import numpy
import sys
import random

from htm.bindings.sdr import SDR, Metrics
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.algorithms import Classifier

class BallAgent:

    motorDimensions = 3

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

        self.encodingWidth = self.resolutionX * self.resolutionY

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

        self.centerX = 0
        self.centerY = 0

        self.predPositions = []
        self.lastTwo = [ [ 0, 0 ] ] * 2

        self.sequenceLength = 0

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

    def EncodeSenseData ( self, bitRepresentation):
    # Encodes sense data as an SDR and returns it.

        encoding = bitRepresentation
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
        if self.Within( self.screenWidth / 2, centerX - ( localDimX / 2 ), centerX + ( localDimX / 2 ), False ):
            for y in range( self.resolutionY ):
                bitToAdd = int( ( ( self.screenWidth / 2 ) - centerX + ( localDimX / 2 ) ) / scaleX ) + ( y * self.resolutionX )
                if bitToAdd >= self.encodingWidth or bitToAdd < 0:
                    sys.exit( "Right side border bit outside encodingWidth size." )
                else:
                    localBitRep.append( bitToAdd )

        # Left side border bits.
        if self.Within( -self.screenWidth / 2, centerX - ( localDimX / 2 ), centerX + ( localDimX / 2 ), False ):
            for y in range( self.resolutionY ):
                bitToAdd = int( ( -( self.screenWidth / 2 ) - centerX + ( localDimX / 2 ) ) / scaleX ) + ( y * self.resolutionX )
                if bitToAdd >= self.encodingWidth or bitToAdd < 0:
                    sys.exit( "Left side border bit outside encodingWidth size." )
                else:
                    localBitRep.append( bitToAdd )

        # Top side border bits.
        if self.Within( self.screenHeight / 2, centerY - ( localDimY / 2 ), centerY + ( localDimY / 2 ), False ):
            for x in range( self.resolutionX ):
                bitToAdd = x + ( int( ( (self.screenHeight / 2 ) - centerY + ( localDimY / 2 ) ) / scaleY ) * self.resolutionX )
                if bitToAdd >= self.encodingWidth or bitToAdd < 0:
                    sys.exit( "Top border bit outside encodingWidth size:" )
                else:
                    localBitRep.append( bitToAdd )

        # Bottom side border bits.
        if self.Within( -self.screenHeight / 2, centerY - ( localDimY / 2 ), centerY + ( localDimY / 2 ), False ):
            for x in range( self.resolutionX ):
                bitToAdd = x + ( int( ( -(self.screenHeight / 2 ) - centerY + ( localDimY / 2 ) ) / scaleY ) * self.resolutionX )
                if bitToAdd >= self.encodingWidth or bitToAdd < 0:
                    sys.exit( "Bottom border bit outside encodingWidth size." )
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
                        if bitToAdd >= self.encodingWidth or bitToAdd < 0:
                            sys.exit( "Ball bit outside encodingWidth size." )
                        else:
                            localBitRep.append( bitToAdd )

#        print("----------------------------------------------------------------")

#        whatPrintX   = self.resolutionX
#        whatPrintY   = self.resolutionY
#        whatPrintRep = localBitRep

#        for y in range( whatPrintY ):
#            for x in range( whatPrintX ):
#                if x == whatPrintX - 1:
#                    print ("ENDO")
#                    endRep = "\n"
#                else:
#                    endRep = ""
#                    if x + (y * whatPrintX ) in whatPrintRep:
#                        print(1, end=endRep)
#                    else:
#                        print(0, end=endRep)

        bitRepSDR = SDR( self.encodingWidth )
        bitRepSDR.sparse = numpy.unique( localBitRep )
        return bitRepSDR

    def PredictTimeStep ( self ):
    # Use last two ball positions and tp to predict next number of ball positions.

        self.tp.reset()
        self.predPositions.clear()

        tempCenterX = self.lastTwo[ 0 ][ 0 ]
        tempCenterY = self.lastTwo[ 0 ][ 1 ]
        tempBallX   = self.lastTwo[ 1 ][ 0 ]
        tempBallY   = self.lastTwo[ 1 ][ 1 ]

        # Feed in last two ball positions, without learning, to start prediction.
        senseSDR = self.EncodeSenseData( self.BuildLocalBitRep( self.localDimX, self.localDimY, tempCenterX, tempCenterY, self.lastTwo[ 0 ][ 0 ], self.lastTwo[ 0 ][ 0 ] ) )
        self.tp.compute( senseSDR, learn = False )
        self.tp.activateDendrites( learn = False )
        senseSDR = self.EncodeSenseData( self.BuildLocalBitRep( self.localDimX, self.localDimY, tempCenterX, tempCenterY, self.lastTwo[ 1 ][ 0 ], self.lastTwo[ 1 ][ 0 ] ) )
        self.tp.compute( senseSDR, learn = False )
        self.tp.activateDendrites( learn = False )
        activeCellsTP = self.tp.getActiveCells()
        predictCellsTP = self.tp.getPredictiveCells()
        shiftX = numpy.argmax( self.xPosition.infer( pattern = activeCellsTP ) ) - 100
        shiftY = numpy.argmax( self.yPosition.infer( pattern = activeCellsTP ) ) - 100
        tempBallX += shiftX
        tempBallY += shiftY

        # Predict next number of ball positions.
        for predSteps in range( 10 ):

            self.predPositions.append( [ tempBallX, tempBallY ] )

            if not self.Within( tempBallX, tempCenterX - ( self.localDimX / 2 ), tempCenterX + ( self.localDimX / 2 ), True ) or not self.Within( tempBallY, tempCenterY - ( self.localDimY / 2 ), tempCenterY + ( self.localDimY / 2 ), True ):
                tempCenterX = tempBallX
                tempCenterY = tempBallY

            senseSDR = self.EncodeSenseData( self.BuildLocalBitRep( self.localDimX, self.localDimY, tempCenterX, tempCenterY, tempBallX, tempBallY ) )
            self.tp.compute( senseSDR, learn = False )
            self.tp.activateDendrites( learn = False )
            activeCellsTP = self.tp.getActiveCells()
            predictCellsTP = self.tp.getPredictiveCells()
            shiftX = numpy.argmax( self.xPosition.infer( pattern = activeCellsTP ) ) - 100
            shiftY = numpy.argmax( self.yPosition.infer( pattern = activeCellsTP ) ) - 100
            tempBallX += shiftX
            tempBallY += shiftY

    def Brain ( self, ballX, ballY ):
    # Agents brain center.

        # Update last three observed ball positions, used for prediction system.
        self.lastTwo.pop( 0 )
        self.lastTwo.append( [ ballX, ballY ] )

        resetOrNo = random.randint( 1, 15 )
        if resetOrNo <= self.sequenceLength:
            self.PredictTimeStep()
            self.sequenceLength = 0
            self.tp.reset()

        self.sequenceLength += 1

        # If ball has moved out of attention view then shift attention view to ball.
        if not self.Within( ballX, self.centerX - ( self.localDimX / 2 ), self.centerX + ( self.localDimX / 2 ), True ) or not self.Within( ballY, self.centerY - ( self.localDimY / 2 ), self.centerY + ( self.localDimY / 2 ), True ):
            self.centerX = ballX
            self.centerY = ballY

        # Feed in attention view to encode bit representation and produce SDR.
        senseSDR = self.EncodeSenseData( self.BuildLocalBitRep( self.localDimX, self.localDimY, self.centerX, self.centerY, ballX, ballY ) )

        # Feed SDR into tp to learn as part of sequence.
        self.tp.compute( senseSDR, learn = True )
        self.tp.activateDendrites( learn = True )
        activeCellsTP = self.tp.getActiveCells()

        # Learn the active cell encoding as a ball shift (include a shift since always must be positive).
        self.xPosition.learn( pattern = activeCellsTP, classification = self.lastTwo[ 1 ][ 0 ] - self.lastTwo[ 0 ][ 0 ] + 100 )
        self.yPosition.learn( pattern = activeCellsTP, classification = self.lastTwo[ 1 ][ 1 ] - self.lastTwo[ 0 ][ 1 ] + 100 )

#        self.LearnTimeStep( self.secondLastData, self.lastData, [ ballX, ballY ], True )

        # Set present ball coordinates for next time-step.
#        self.secondLastData = [ self.lastData[ 0 ], self.lastData[ 1 ] ]
#        self.lastData = [ ballX, ballY ]

        # Store last and present locations.
#        predPositions = []
#        predPositions.append( self.secondLastData )
#        predPositions.append( self.lastData )

        # Predict next 10 time step locations and store them.
#        for step in range( 10 ):
#            nextPosition = self.PredictTimeStep( predPositions[ -2 ], predPositions[ -1 ], False )
#            if self.Within( nextPosition[ 0 ], -self.screenWidth / 2, self.screenWidth / 2 ) and self.Within( nextPosition[ 1 ], -self.screenHeight / 2, self.screenHeight / 2 ):
#                predPositions.append( nextPosition )
#            else:
#                break
