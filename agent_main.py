import numpy
from useful_functions import Within, BinarySearch, NoRepeatInsort
from random import randrange

from htm.bindings.sdr import SDR, Metrics
import htm.bindings.encoders
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from vector_memory import VectorMemory

class Agent:

    def __init__( self, name, senseDimX, senseDimY, screenHeight, screenWidth, ballWidth, ballHeight, paddleWidth, paddleHeight ):

        self.ID = name
        self.senseResX = 20
        self.senseResY = 20
        self.scaleX = senseDimX / self.senseResX
        self.scaleY = senseDimY / self.senseResY
        if self.scaleX < 1 or self.scaleY < 1:
            sys.exit( "Resolution (X and Y) must be less than local dimensions (X and Y)." )

        self.screenHeight = screenHeight
        self.screenWidth  = screenWidth
        self.ballHeight   = ballHeight
        self.ballWidth    = ballWidth
        self.paddleHeight = paddleHeight
        self.paddleWidth  = paddleWidth

        self.encodingWidth = self.senseResX * self.senseResY

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
            boostStrength              = 0.0,
            seed                       = -1,
            wrapAround                 = False
        )

#        self.tp = TemporalMemory(
#            columnDimensions          = ( 2048, ),
#            cellsPerColumn            = 32,
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

        vpsVectorDim = 2
        self.vp = VectorMemory(
            columnDimensions          = 2048,
            cellsPerColumn            = 4,
            numObjectCells            = 1000,
            FActivationThresholdMin   = 25,
            FActivationThresholdMax   = 30,
            initialPermanence         = 0.3,
            permanenceIncrement       = 0.04,
            permanenceDecrement       = 0.005,
            permanenceDecay           = 0.001,
            segmentDecay              = 99999,
            objectRepActivation       = 25,
            maxSynapsesToAddPer       = 5,
            maxSynapsesPerSegment     = 50,
            equalityThreshold         = 35,
            vectorDimensions          = vpsVectorDim,
            initialVectorScaleFactor  = 0.05,
            initVectorConfidence      = 0.3,
        )

        self.lastVector = []
        self.newVector  = []
        for i in range( vpsVectorDim ):
            self.lastVector.append( 0 )
            self.newVector.append( 0 )

        self.senseX = 0
        self.senseY = 0

        self.localBitRep = []

    def ReturnSenseOrganLocation( self ):
    # Return the senseX and senseY.

        return self.senseX, self.senseY

    def GetLogData( self ):
    # Get the local log data and return it.

        log_data = []

        log_data.append( self.PrintBitRep( self.localBitRep, self.senseResX, self.senseResY, self.senseX, self.senseY ) )

        log_data.append( "Last Vector: " + str( self.lastVector ) + ", New Vector: " + str( self.newVector ) )

        self.vp.BuildLogData( log_data )

        return log_data

    def SendStateData( self, stateNumber ):
    # Get the active cells from vp

        return self.vp.SendData( stateNumber )

    def GetStateData( self ):
    # Get the state data from vp.

        return self.vp.GetStateInformation()

    def GetGraphData( self ):
    # Return the number of active cells in vp in this time step.

        return self.vp.GetGraphData()

    def PrintBitRep( self, whatPrintRep, whatPrintX, whatPrintY, centerX, centerY ):
    # Returns the bit represention as a string.

        bitRepString = ""

        bitRepString += str( self.ID ) + ", CenterX: " + str( centerX ) + ", CenterY: " + str( centerY ) + "\n"
        bitRepString += "ResolutionX: " + str( whatPrintX ) + ", ResolutionY: " + str( whatPrintY ) + "\n"

        for y in range( whatPrintY ):
            for x in range( whatPrintX ):
                if x == whatPrintX - 1:
                    bitRepString += "\n"
                else:
                    if BinarySearch( whatPrintRep, x + ( y * whatPrintX ) ):
                        bitRepString += "1"
                    else:
                        bitRepString += "0"

        bitRepString += "\n"

        return bitRepString

    def BuildLocalBitRep( self, paddleAY, paddleAX, paddleBY, paddleBX, ballX, ballY ):
    # Builds a bit-rep SDR of localDim dimensions centered around point with resolution.

        maxArraySize = self.senseResX * self.senseResY

        self.localBitRep = []

        for x in range( self.senseResX ):
            for y in range( self.senseResY ):
                posX = int( ( ( x - ( self.senseResX / 2 ) ) * self.scaleX ) + ( self.scaleX / 2 ) + self.senseX )
                posY = int( ( ( y - ( self.senseResY / 2 ) ) * self.scaleY ) + ( self.scaleY / 2 ) + self.senseY )

                # Ball bits.
                if Within( posX, ballX - self.ballWidth, ballX + self.ballWidth, True ) and Within( posY, ballY - self.ballHeight, ballY + self.ballHeight, True ):
                    NoRepeatInsort( self.localBitRep, x + ( y * self.senseResX ) )

                # Wall bits.
                elif not Within( posX, -self.screenWidth / 2, self.screenWidth / 2, True ) or not Within( posY, -self.screenHeight / 2, self.screenHeight / 2, True ):
                    NoRepeatInsort( self.localBitRep, x + ( y * self.senseResX ) )

                # Paddle A bits.
                elif Within( posX, paddleAX - self.paddleWidth, paddleAX + self.paddleWidth, True ) and Within( posY, paddleAY - self.paddleHeight, paddleAY + self.paddleHeight, True ):
                    NoRepeatInsort( self.localBitRep, x + ( y * self.senseResX ) )

                # Paddle A bits.
                elif Within( posX, paddleBX - self.paddleWidth, paddleBX + self.paddleWidth, True ) and Within( posY, paddleBY - self.paddleHeight, paddleBY + self.paddleHeight, True ):
                    NoRepeatInsort( self.localBitRep, x + ( y * self.senseResX ) )

#        print( self.PrintBitRep( self.localBitRep, self.senseResX, self.senseResY, self.senseX, self.senseY ) )

        bitRepSDR = SDR( maxArraySize )
        bitRepSDR.sparse = numpy.unique( self.localBitRep )
        return bitRepSDR

    def EncodeSenseData ( self, paddleAY, paddleAX, paddleBY, paddleBX, ballX, ballY ):
    # Encodes sense data as an SDR and returns it.

        encoding = self.BuildLocalBitRep( paddleAY, paddleAX, paddleBY, paddleBX, ballX, ballY )

        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    def Brain ( self, paddleAX, paddleAY, paddleBX, paddleBY, ballX, ballY, ballVelX, ballVelY ):
    # Agents brain center.

        self.senseX = self.senseX + self.newVector[ 0 ]
        self.senseY = self.senseY + self.newVector[ 1 ]

        senseSDR = self.EncodeSenseData( paddleAY, paddleAX, paddleBY, paddleBX, ballX, ballY )

        self.lastVector = self.newVector.copy()

        # Feed in input and lastVector into vector memory network.
        self.vp.Compute( senseSDR, self.lastVector )

        # Generate random motion vector for next time step.
        senseOrganLocation = randrange( 3 )
        if senseOrganLocation == 0:
            newSenseLocation = ( paddleAX, paddleAY )
        elif senseOrganLocation == 1:
            newSenseLocation = ( paddleBX, paddleBY )
        elif senseOrganLocation == 2:
            newSenseLocation = ( ballX, ballY )
        self.newVector = [ newSenseLocation[ 0 ] - self.senseX, newSenseLocation[ 1 ] - self.senseY ]

        self.vp.PredictFCells( self.newVector )
