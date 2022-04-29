import sys
import numpy
from random import randrange, sample
from bisect import bisect_left
from useful_functions import Within

from htm.bindings.sdr import SDR, Metrics
import htm.bindings.encoders
ScalarEncoder = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import Classifier
from vector_memory import VectorMemory

class AgentOrange:

    def __init__( self, name, resX, resY, spatialDimensions, otherDimensions ):

        self.ID = name

        self.resolutionX = resX
        self.resolutionY = resY

        # Set up encoder parameters
        colourEncodeParams = ScalarEncoderParameters()

        colourEncodeParams.activeBits = 5
        colourEncodeParams.radius     = 1
        colourEncodeParams.clipInput  = False
        colourEncodeParams.minimum    = 0
        colourEncodeParams.maximum    = 3
        colourEncodeParams.periodic   = False

        self.colourEncoder = ScalarEncoder( colourEncodeParams )

        self.encodingWidth = self.resolutionX * self.resolutionY * 4

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

        vpsVectorDim = spatialDimensions + otherDimensions
        self.vp = VectorMemory(
            columnDimensions          = 2048,
            cellsPerColumn            = 4,
            FActivationThresholdMin   = 30,
            FActivationThresholdMax   = 30,
            initialPermanence         = 0.3,
            permanenceIncrement       = 0.04,
            permanenceDecrement       = 0.005,
            permanenceDecay           = 0.001,
            segmentDecay              = 99999,
            objectRepActivation       = 25,
            numObjectCells            = 1000,
            OCellActivationThreshold  = 10,
            maxSynapsesToAddPer       = 5,
            maxSynapsesPerSegment     = 50,
            equalityThreshold         = 35,
            vectorDimensions          = vpsVectorDim,
            initialVectorScaleFactor  = 0.005,
            initVectorConfidence      = 0.3,
            vectorScoreThreshold      = 0.1,
            WMEntryDecay              = 25,
            WMEntrySize               = 3,
            WMStabilityPct            = 1.0,
        )

        self.lastVector = []
        self.newVector  = []
        for i in range( vpsVectorDim ):
            self.lastVector.append( 0 )
            self.newVector.append( 0 )
        self.lastColour = 1

        # Stats for end report.
        self.top_left     = []
        self.top_right    = []
        self.bottom_left  = []
        self.bottom_right = []

        self.localBitRep  = []
        self.centerX      = 0
        self.centerY      = 0

        # For Reflection.
        self.stateReflectTime   = 0
        self.timeToBeReflective = 100
        self.lastWMEntryID = 0
        self.thisWMEntryID = 0
        self.nextWMEntryID = 0

    def GetLogData( self ):
    # Get the local log data and return it.

        log_data = []

        self.PrintBitRep( log_data )

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

    def PrintBitRep( self, log_data ):
    # Returns the bit represention as a string.

        log_data.append( "CenterX: " + str( self.centerX ) + ", CenterY: " + str( self.centerY ) )

        for y in range( self.resolutionY ):
            log_input = ""
            for x in range( self.resolutionX ):
                    if x + ( y * self.resolutionX ) in self.localBitRep:
                        log_input = log_input + str( 0 )
                    elif ( x + ( y * self.resolutionX ) ) + ( self.resolutionX * self.resolutionY ) in self.localBitRep:
                        log_input = log_input + str( 1 )
                    elif ( x + ( y * self.resolutionX ) ) + ( 2 * self.resolutionX * self.resolutionY ) in self.localBitRep:
                        log_input = log_input + str( 2 )
                    elif ( x + ( y * self.resolutionX ) ) + ( 3 * self.resolutionX * self.resolutionY ) in self.localBitRep:
                        log_input = log_input + str( 3 )
            log_data.append( log_input )

    def BuildLocalBitRep( self, centerX, centerY, objX, objY, objW, objH, objC, noisePct ):
    # Builds a bit-rep SDR of localDim dimensions centered around point with resolution.
    # centerX and centerY is the center point of our vision field.
    # objX and objY: origin point of the object we are examining, objC: object colour, objW and objH: height and width.

        maxArraySize = self.resolutionX * self.resolutionY * 4

        self.localBitRep = []
        self.centerX     = centerX
        self.centerY     = centerY

        # Object bits.
        for x in range( self.resolutionX ):
            for y in range( self.resolutionY ):
                posX = x - ( self.resolutionX / 2 ) + centerX
                posY = y - ( self.resolutionY / 2 ) + centerY
                if Within( posX, objX - objW, objX + objW, True ) and Within( posY, objY - objH, objY + objH, True ):
                    self.localBitRep.append( x + ( y * self.resolutionX ) + (objC * self.resolutionX * self.resolutionY ) )
                else:
                    self.localBitRep.append( x + ( y * self.resolutionX ) )

        self.localBitRep.sort()

        # Add noise.
        if noisePct > 0.0:
            noiseIndices = sample( range( maxArraySize ), int( noisePct * maxArraySize ) )
            for n in noiseIndices:
                index = bisect_left( self.localBitRep, n )
                if index != len( self.localBitRep ) and self.localBitRep[ index ] == n:
                    del self.localBitRep[ index ]
                else:
                    self.localBitRep.insert( index, n )

        bitRepSDR = SDR( maxArraySize )
        bitRepSDR.sparse = numpy.unique( self.localBitRep )
        return bitRepSDR

    def EncodeSenseData ( self, sensePosX, sensePosY, objX, objY, objW, objH, objC, noisePct ):
    # Get sensory information and encode it as an SDR in the sense network.

        # Encode colour
        objBits = self.BuildLocalBitRep( sensePosX, sensePosY, objX, objY, objW, objH, objC, noisePct )

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = objBits
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    def Reflect( self ):
    # Go through a reflective period to learn important stored states uniquly in memory.
    # Return True if Agent still needs time to reflect, and False if it is done.

        # When we start a new entry begin by refreshing vector memory.
        if self.stateReflectTime == 0:
            self.vp.Refresh()

            self.lastWMEntryID = 0
            self.thisWMEntryID = 0
            self.nextWMEntryID = 0

            # Get Vector Memory to generate unique cell reps for each entry.
            self.vp.GenerateUniqueCellReps()

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

    def Brain ( self, objX, objY, objW, objH, objC, sensePosX, sensePosY, noisePct ):
    # The central brain function of the agent.

        # Encode the input column SDR for current position.
        senseSDR = self.EncodeSenseData( sensePosX, sensePosY, objX, objY, objW, objH, objC, noisePct )

        # Generate the Agent's motion vector.
        self.lastVector = self.newVector.copy()

        # Generate random motion of the sense organ.
        whichPos = randrange( 4 )
        chosePos = [ 100, 100 ]
        if whichPos == 0:
            chosePos[ 0 ] = 100
            chosePos[ 1 ] = 100
        elif whichPos == 1:
            chosePos[ 0 ] = -100
            chosePos[ 1 ] = -100
        elif whichPos == 2:
            chosePos[ 0 ] = -100
            chosePos[ 1 ] = 100
        elif whichPos == 3:
            chosePos[ 0 ] = 100
            chosePos[ 1 ] = -100
        self.newVector[ 0 ] = chosePos[ 0 ] - sensePosX
        self.newVector[ 1 ] = chosePos[ 1 ] - sensePosY
        # Append on the colour portion of the vector.
        self.newVector[ 2 ] = ( objC - self.lastColour ) * 100
        self.lastColour = objC

        # Compute cell activation and generate next predicted cells.
        if self.newVector[ 2 ] != 0:
            feelingState = 1.0
        else:
            feelingState = self.vp.CheckOCellFeeling()

        if feelingState != 0.0:
            self.vp.Memorize( feelingState )

        self.vp.Compute( senseSDR, self.lastVector )

        # Use FSegments to predict next set of inputs ( put the cells into predictive states ), given newVector.
        self.vp.PredictFCells( self.newVector )

        return self.newVector
