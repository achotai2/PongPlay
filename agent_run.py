import sys
import numpy
from random import randrange, sample, getrandbits
from bisect import bisect_left
from useful_functions import Within, BinarySearch, DelIfIn, NoRepeatInsort
from new_vector_memory import NewVectorMemory

from htm.bindings.sdr import SDR, Metrics
import htm.bindings.encoders
ScalarEncoder = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import Classifier

class AgentRun:

    def __init__( self, name, senseResX, senseResY, spatialDimensions, otherDimensions ):

        self.ID = name

        # Set up eye input encoding parameters.
        self.resolutionX = senseResX
        self.resolutionY = senseResY
        self.numColours  = 3                                                    # Red, Green, Blue

        self.eyeInputEncodingWidth  = self.resolutionX * self.resolutionY * self.numColours

        # Set up vector encoder parameters.
        XEncodeParams = ScalarEncoderParameters()
        XEncodeParams.activeBits = 5
        XEncodeParams.radius     = 10
        XEncodeParams.clipInput  = True
        XEncodeParams.minimum    = -200
        XEncodeParams.maximum    = 200
        XEncodeParams.periodic   = False
        self.XTransformEncoder = ScalarEncoder( XEncodeParams )
        YEncodeParams = ScalarEncoderParameters()
        YEncodeParams.activeBits = 5
        YEncodeParams.radius     = 10
        YEncodeParams.clipInput  = True
        YEncodeParams.minimum    = -200
        YEncodeParams.maximum    = 200
        YEncodeParams.periodic   = False
        self.YTransformEncoder = ScalarEncoder( YEncodeParams )

        self.transformEncodingWidth = self.XTransformEncoder.size + self.YTransformEncoder.size

        vpsVectorDim = spatialDimensions + otherDimensions

        # The dictionary for defining useful terms.
        self.termsDict = {
            "columnDimensions"          : 2048,                 # Dimensions of the column space.
            "cellsPerColumn"            : 4,                    # Number of cells per column.
            "FActivationThresholdMin"   : 30,                   # Min threshold of active connected incident synapses...
            "FActivationThresholdMax"   : 30,                   # Max threshold of active connected incident synapses...# needed to activate segment.
            "initialPermanence"         : 0.3,                  # Initial permanence of a new synapse.
            "permanenceIncrement"       : 0.05,                 # Amount by which permanences of synapses are incremented during learning.
            "permanenceDecrement"       : 0.01,                # Amount by which permanences of synapses are decremented during learning.
            "permanenceDecay"           : 0.001,                # Amount to decay permances each time step if < 1.0.
            "segmentDecay"              : -1,                  # If a segment hasn't been active in this many time steps then delete it. If -1 then there is none.
            "maxTotalSegments"          : -1,                 # Caps the maximum segments that can exist in a network. If -1 then there is no limit.
            "segStimulatedDecay"        : 0.2,                 # The rate at which segment stimulation decays from 1.0 to 0.0
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
            "permanenceLowerThreshold"  : 0.1,                  # The lower threshold for synapses.
            "maxSequenceLength"         : 1,                    # The length of the cell context sequences in vector memory.
        }

        # The eye input pooler.
        self.sp = SpatialPooler(
            inputDimensions            = ( self.eyeInputEncodingWidth, ),
            columnDimensions           = ( 2048, ),
            potentialPct               = 0.85,
            potentialRadius            = self.eyeInputEncodingWidth,
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

        # The transformation input pooler.
        self.trp = SpatialPooler(
            inputDimensions            = ( self.transformEncodingWidth, ),
            columnDimensions           = ( 2048, ),
            potentialPct               = 0.85,
            potentialRadius            = self.transformEncodingWidth,
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

        self.vp = NewVectorMemory( self.termsDict )

        self.lastVector = []
        self.newVector  = []
        for i in range( vpsVectorDim ):
            self.lastVector.append( 0 )
            self.newVector.append( 0 )

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

            for c in range( self.numColours ):
                for x in range( self.resolutionX ):
                    if BinarySearch( self.localBitRep, ( x + ( y * self.resolutionX ) ) + ( c * self.resolutionX * self.resolutionY ) ):
                        log_input = log_input + str( 1 )
                    else:
                        log_input = log_input + str( 0 )

                log_input = log_input + "\t"

            log_data.append( log_input )

    def BuildLocalBitRep( self, centerX, centerY, objectList, noisePct ):
    # Builds a bit-rep SDR of localDim dimensions centered around point with resolution.
    # centerX and centerY is the center point of our vision field.
    # objX and objY: origin point of the object we are examining, objC: object colour, objW and objH: height and width.

        self.localBitRep = []
        self.centerX     = centerX
        self.centerY     = centerY

        # Object bits.
        for object in objectList:
            # Safety check on colour value.
            if object[ 4 ] >= self.numColours:
                print( "BuildLocalBitRep(): Object colour value passed (" + str( object[ 4 ] ) + ") was greater than number of colour receptors (" + str( self.numColours ) + ")." )
                exit()

            for x in range( self.resolutionX ):
                for y in range( self.resolutionY ):
                    posX = x - ( self.resolutionX / 2 ) + centerX
                    posY = y - ( self.resolutionY / 2 ) + centerY
                    if Within( posX, object[ 0 ] - object[ 2 ], object[ 0 ] + object[ 2 ], True ) and Within( posY, object[ 1 ] - object[ 3 ], object[ 1 ] + object[ 3 ], True ):
                        self.localBitRep.append( x + ( y * self.resolutionX ) + ( object[ 4 ] * self.resolutionX * self.resolutionY ) )

        self.localBitRep.sort()

        # Add noise.
        if noisePct > 0.0:
            noiseIndices = sample( range( self.eyeInputEncodingWidth ), int( noisePct * self.eyeInputEncodingWidth ) )

            for n in noiseIndices:
                bitOn = bool( getrandbits( 1 ) )
                if bitOn:
                    NoRepeatInsort( self.localBitRep, n )
                else:
                    DelIfIn( self.localBitRep, n )

        bitRepSDR = SDR( self.eyeInputEncodingWidth )
        bitRepSDR.sparse = numpy.unique( self.localBitRep )

        return bitRepSDR

    def EncodeSenseData ( self, sensePosX, sensePosY, objectList, noisePct ):
    # Get sensory information and encode it as an SDR in the sense network.

        # Encode colour
        objBits = self.BuildLocalBitRep( sensePosX, sensePosY, objectList, noisePct )

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = objBits
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    def EncodeTransformationData( self, vector ):
    # Get vector and encode it as an SDR.

        XBits = self.XTransformEncoder.encode( vector[ 0 ] )
        YBits = self.YTransformEncoder.encode( vector[ 1 ] )

        encoding     = SDR( self.transformEncodingWidth ).concatenate( [ XBits, YBits ] )
        transformSDR = SDR( self.trp.getColumnDimensions() )
        self.trp.compute( encoding, True, transformSDR )

        return transformSDR

    def Brain ( self, objectList, sensePosX, sensePosY, noisePct, randomInput ):
    # The central brain function of the agent.

        if randomInput:
            noisePct = 1.00

        # Encode the input column SDR for current position.
        senseSDR = self.EncodeSenseData( sensePosX, sensePosY, objectList, noisePct )

        # Generate the Agent's motion vector.
        self.lastVector = self.newVector.copy()

        # Generate random motion of the sense organ.
        whichObj       = randrange( len( objectList ) )
        chosePos       = [ objectList[ whichObj ][ 0 ], objectList[ whichObj ][ 1 ] ]
        self.newVector[ 0 ] = chosePos[ 0 ] - sensePosX
        self.newVector[ 1 ] = chosePos[ 1 ] - sensePosY

        # Encode the transformation vector and get column SDR.
        newVectorSDR = self.EncodeTransformationData( self.newVector ).sparse.tolist()

        # Compute the action of vector memory, and learn on the synapses.
        self.vp.Compute( senseSDR, newVectorSDR )

        return self.newVector.copy()
