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

        self.encodingWidth = ( self.senseResX * self.senseResY ) + 100

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
            boostStrength              = 1.0,
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

        vectorMemoryDict = {
            "columnDimensions"          : 2048,                 # Dimensions of the column space.
            "cellsPerColumn"            : 4,                    # Number of cells per column.
            "FActivationThresholdMin"   : 30,                   # Min threshold of active connected incident synapses...
            "FActivationThresholdMax"   : 30,                   # Max threshold of active connected incident synapses...# needed to activate segment.
            "initialPermanence"         : 0.3,                  # Initial permanence of a new synapse.
            "permanenceIncrement"       : 0.04,                 # Amount by which permanences of synapses are incremented during learning.
            "permanenceDecrement"       : 0.005,                # Amount by which permanences of synapses are decremented during learning.
            "permanenceDecay"           : 0.001,                # Amount to decay permances each time step if < 1.0.
            "segmentDecay"              : 99999,                # If a segment hasn't been active in this many time steps then delete it.
            "objectRepActivation"       : 25,                   # Number of cells in the Object level.
            "numObjectCells"            : 1000,                 # Number of active OCells in object layer at one time.
            "OCellActivationThreshold"  : 10,                   # Number of segments stimulated activating an OCell for it to become active.
            "maxSynapsesToAddPer"       : 5,                    # The maximum number of FToFSynapses added to a segment during creation.
            "maxSynapsesPerSegment"     : 50,                   # Maximum number of active synapses allowed on a segment.
            "equalityThreshold"         : 35,                   # The number of equal synapses for two segments to be considered identical.
            "vectorDimensions"          : vpsVectorDim,         # The number of dimensions of our vector space.
            "initialStandardDeviation"  : 100,                  # Scales the distance a vector can activate a segment. Smaller means more activation distance.
            "initialVectorConfidence"   : 1.0,                  # The max center score for a vector activating a segment.
            "vectorScoreLowerThreshold" : 0.001,                # The minimum vector score needed to activate a segment.
            "vectorScoreUpperThreshold" : 0.9,
            "vectorConfidenceShift"     : 0.04,
            "standardDeviationShift"    : 1,
            "WMEntryDecay"              : 5,
            "WMEntrySize"               : 3,
            "WMStabilityPct"            : 1.0,
            "vectorRange"               : 1600,
            "numVectorSynapses"         : 100,
            "vectorSynapseScaleFactor"  : 0.8,
            "maxVectorSynapseRadius"    : 15,
        }

        self.vp = VectorMemory( vectorMemoryDict )

        self.lastVector = []
        self.newVector  = []
        for i in range( vpsVectorDim ):
            self.lastVector.append( 0 )
            self.newVector.append( 0 )

        self.senseX = 0
        self.senseY = 0

        self.localBitRep = []

        # For Reflection.
        self.stateReflectTime   = 0
        self.timeToBeReflective = 100
        self.lastWMEntryID = 0
        self.thisWMEntryID = 0
        self.nextWMEntryID = 0

    def ReturnSenseOrganLocation( self ):
    # Return the senseX and senseY.

        return self.senseX, self.senseY

    def GetLogData( self ):
    # Get the local log data and return it.

        log_data = []

        log_data.append( self.PrintBitRep( self.localBitRep, self.senseResX, self.senseResY, self.senseX, self.senseY ) )

        log_data.append( "Last Vector: " + str( self.lastVector ) + ", New Vector: " + str( self.newVector ) )

        self.vp.BuildLogData( log_data )

        if self.stateReflectTime == 0:
            reflective = False
        else:
            reflective = True

        return log_data, reflective

    def SendStateData( self, stateNumber ):
    # Get the active cells from vp

        return self.vp.SendData( stateNumber )

    def GetStateData( self ):
    # Get the state data from vp.

        return self.vp.GetStateInformation()

    def GetGraphData( self ):
    # Return the number of active cells in vp in this time step.

        if self.stateReflectTime == 0:
            reflective = False
        else:
            reflective = True

        return self.vp.GetGraphData(), reflective

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

#    def BuildLocalBitRep( self, paddleAY, paddleAX, paddleBY, paddleBX, ballX, ballY ):
    def BuildLocalBitRep( self, paddleAY, paddleAX, ballVelX, ballVelY, ballX, ballY ):
    # Builds a bit-rep SDR of localDim dimensions centered around point with resolution.

        self.localBitRep = []

        for x in range( self.senseResX ):
            for y in range( self.senseResY ):
                posX = int( ( ( x - ( self.senseResX / 2 ) ) * self.scaleX ) + ( self.scaleX / 2 ) + self.senseX )
                posY = int( ( ( y - ( self.senseResY / 2 ) ) * self.scaleY ) + ( self.scaleY / 2 ) + self.senseY )

                # Ball bits.
                if Within( posX, ballX - self.ballWidth, ballX + self.ballWidth, True ) and Within( posY, ballY - self.ballHeight, ballY + self.ballHeight, True ):
                    NoRepeatInsort( self.localBitRep, x + ( y * self.senseResX ) )
                    if ballVelX < 0:
                        lowerIndex = self.senseResX * self.senseResY
                        for i in range( lowerIndex, lowerIndex + 25 ):
                            NoRepeatInsort( self.localBitRep, i )
                    elif ballVelX > 0:
                        lowerIndex = ( self.senseResX * self.senseResY ) + 25
                        for i in range( lowerIndex, lowerIndex + 25 ):
                            NoRepeatInsort( self.localBitRep, i )
                    if ballVelY < 0:
                        lowerIndex = ( self.senseResX * self.senseResY ) + 50
                        for i in range( lowerIndex, lowerIndex + 25 ):
                            NoRepeatInsort( self.localBitRep, i )
                    elif ballVelY > 0:
                        lowerIndex = ( self.senseResX * self.senseResY ) + 75
                        for i in range( lowerIndex, self.encodingWidth ):
                            NoRepeatInsort( self.localBitRep, i )

                # Wall bits.
                elif not Within( posX, -self.screenWidth / 2, self.screenWidth / 2, True ) or not Within( posY, -self.screenHeight / 2, self.screenHeight / 2, True ):
                    NoRepeatInsort( self.localBitRep, x + ( y * self.senseResX ) )

                # Paddle A bits.
                elif Within( posX, paddleAX - self.paddleWidth, paddleAX + self.paddleWidth, True ) and Within( posY, paddleAY - self.paddleHeight, paddleAY + self.paddleHeight, True ):
                    NoRepeatInsort( self.localBitRep, x + ( y * self.senseResX ) )

#                # Paddle B bits.
#                elif Within( posX, paddleBX - self.paddleWidth, paddleBX + self.paddleWidth, True ) and Within( posY, paddleBY - self.paddleHeight, paddleBY + self.paddleHeight, True ):
#                    NoRepeatInsort( self.localBitRep, x + ( y * self.senseResX ) )

#        print( self.PrintBitRep( self.localBitRep, self.senseResX, self.senseResY, self.senseX, self.senseY ) )

        bitRepSDR = SDR( self.encodingWidth )
        bitRepSDR.sparse = numpy.unique( self.localBitRep )
        return bitRepSDR

#    def EncodeSenseData ( self, paddleAY, paddleAX, paddleBY, paddleBX, ballX, ballY ):
    def EncodeSenseData ( self, paddleAY, paddleAX, ballVelX, ballVelY, ballX, ballY ):
    # Encodes sense data as an SDR and returns it.

#        encoding = self.BuildLocalBitRep( paddleAY, paddleAX, paddleBY, paddleBX, ballX, ballY )
        encoding = self.BuildLocalBitRep( paddleAY, paddleAX, ballVelX, ballVelY, ballX, ballY )

        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    def Reflect( self, feelingState ):
    # Go through a reflective period to learn important stored states uniquly in memory.
    # Return True if Agent still needs time to reflect, and False if it is done.

        # When we start a new entry begin by refreshing vector memory.
        if self.stateReflectTime == 0:
            self.vp.Refresh()

            self.lastWMEntryID = 0
            self.thisWMEntryID = 0
            self.nextWMEntryID = 0

            # Get Vector Memory to generate unique cell reps for each entry.
            self.vp.GenerateUniqueCellReps( feelingState )

        # Change entryIDs.
        self.lastWMEntryID = self.thisWMEntryID
        self.thisWMEntryID = self.nextWMEntryID
        self.nextWMEntryID = self.vp.ChooseNextReflectionEntry()

        # Trigger vector memory to reflect on this columnSDR and vector, generate predicted cells, and perform learning on segments.
        self.vp.VectorMemoryReflect( self.lastWMEntryID, self.thisWMEntryID, self.nextWMEntryID )

        # Update reflection time and check if done reflecting.
        self.stateReflectTime += 1

        if self.stateReflectTime > self.timeToBeReflective:
            self.stateReflectTime = 0
            if not self.vp.DeleteSavedState():
                self.vp.Refresh()
                return False
        return True

#    def Brain ( self, paddleAX, paddleAY, paddleBX, paddleBY, ballX, ballY, ballVelX, ballVelY, feelingState ):
    def Brain ( self, paddleAX, paddleAY, ballX, ballY, ballVelX, ballVelY, feelingState ):
    # Agents brain center.

        self.senseX = self.senseX + self.newVector[ 0 ]
        self.senseY = self.senseY + self.newVector[ 1 ]

#        senseSDR = self.EncodeSenseData( paddleAY, paddleAX, paddleBY, paddleBX, ballX, ballY )
        senseSDR = self.EncodeSenseData( paddleAY, paddleAX, ballVelX, ballVelY, ballX, ballY )

        self.lastVector = self.newVector.copy()

#        if feelingState == 0.0:
#            feelingState = self.vp.CheckOCellFeeling()
        if feelingState != 0.0:
            self.vp.Memorize( feelingState )

        # Feed in input and lastVector into vector memory network.
        self.vp.Compute( senseSDR, self.lastVector )

        # Generate random motion vector for next time step.
        senseOrganLocation = randrange( 4 )
        if senseOrganLocation == 0:
            newSenseLocation = ( paddleAX, paddleAY )
#        elif senseOrganLocation == 1:
#            newSenseLocation = ( paddleBX, paddleBY )
        elif senseOrganLocation == 1:
            newSenseLocation = ( ballX, ballY )
        elif senseOrganLocation == 2:
            newSenseLocation = ( ballX, self.screenHeight / 2 )
        elif senseOrganLocation == 3:
            newSenseLocation = ( ballX, -self.screenHeight / 2 )

        self.newVector = [ newSenseLocation[ 0 ] - self.senseX, newSenseLocation[ 1 ] - self.senseY ]

        self.vp.PredictFCells( self.newVector )
