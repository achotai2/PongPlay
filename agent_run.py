import sys
import numpy
from random import randrange, sample, getrandbits, choice
from bisect import bisect_left
from useful_functions import Within, BinarySearch, DelIfIn, NoRepeatInsort
from new_vector_memory import NewVectorMemory
from classifier import Classifier

from htm.bindings.sdr import SDR, Metrics
import htm.bindings.encoders
ScalarEncoder = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters
from htm.bindings.algorithms import SpatialPooler

class AgentRun:

    def __init__( self, name, senseResX, senseResY, screenWidth, screenHeight ):

        self.ID = name

        # The dictionary for defining useful terms.
        self.termsDict = {
            "columnDimensions"          : 2048,                 # Dimensions of the column space.
            "transformColumnDimensions" : 2048,                 # Dimenions of transformation column space
            "positionColumnDimensions"  : 2048,                 # Dimenions of position column space
#            "potentialPct"              : 0.85,                 # Used by SpatialPooler for what pct of columns are connected to their input.
            "boostStrength"             : 0.5,                  # Spatial memory boostStrength.
            "numActiveColumnsPerInhArea": 40,                   # Number of columns active given input.
            "cellsPerColumn"            : 4,                    # Number of cells per column.
            "FActivationThresholdMin"   : 30,                   # Min threshold of active connected incident synapses...
            "FActivationThresholdMax"   : 30,                   # Max threshold of active connected incident synapses...# needed to activate segment.
            "initialPermanence"         : 0.3,                  # Initial permanence of a new synapse.
            "permanenceIncrement"       : 0.04,                 # Amount by which permanences of synapses are incremented during learning.
            "permanenceDecrement"       : 0.005,                 # Amount by which permanences of synapses are decremented during learning.
            "permanenceDecay"           : 0.001,                # Amount to decay permances each time step if < 1.0.
            "permanenceLowerThreshold"  : 0.1,                  # The lower threshold for synapses.
            "maxSynapsesToAddPer"       : 1,                    # The maximum number of incident synapses added to a segment during learning.
            "maxSynapsesPerSegment"     : 100,                  # Maximum number of incident synapses allowed on a segment.
            "maxIncidentOnCell"         : 10,                   # The maximum number of segments that cell can be incident to.
            "maxTotalSegments"          : 5000,                 # The maximum number of segments allowed in the network.
            "confidenceConfident"       : 0.8,                  # The confidenceScore above which we consider the segment is a good prediction.
            "numTransformDimensions"    : 4,                    # The number of dimensions in the transform vector.
        }

        # Set up eye input encoding parameters.
        self.resolutionX = senseResX
        self.resolutionY = senseResY
        self.numColours  = 3                                                    # Red, Green, Blue
        self.eyeInputEncodingWidth  = self.resolutionX * self.resolutionY * self.numColours

        # The eye input pooler.
        self.sp = SpatialPooler(
            inputDimensions            = ( self.eyeInputEncodingWidth, ),
            columnDimensions           = ( self.termsDict[ "columnDimensions" ], ),
            potentialPct               = 0.85,
            potentialRadius            = self.eyeInputEncodingWidth,
            globalInhibition           = True,
            localAreaDensity           = 0,
            numActiveColumnsPerInhArea = self.termsDict[ "numActiveColumnsPerInhArea" ],
            synPermInactiveDec         = 0.005,
            synPermActiveInc           = 0.04,
            synPermConnected           = 0.1,
            boostStrength              = self.termsDict[ "boostStrength" ],
            seed                       = -1,
            wrapAround                 = False
        )

        # Set up vector encoder parameters.
        vectorEncodeParams = ScalarEncoderParameters()
        vectorEncodeParams.activeBits = 5
        vectorEncodeParams.radius     = 10
        vectorEncodeParams.clipInput  = True
        vectorEncodeParams.minimum    = -200
        vectorEncodeParams.maximum    = 200
        vectorEncodeParams.periodic   = False
        self.xEyeEncoder  = ScalarEncoder( vectorEncodeParams )
        self.yEyeEncoder  = ScalarEncoder( vectorEncodeParams )
        self.xMoveEncoder = ScalarEncoder( vectorEncodeParams )
        self.yMoveEncoder = ScalarEncoder( vectorEncodeParams )
        self.transformEncodingWidth = self.xEyeEncoder.size + self.yEyeEncoder.size + self.xMoveEncoder.size + self.yMoveEncoder.size

        # The transformation input pooler.
        self.trp = SpatialPooler(
            inputDimensions            = ( self.transformEncodingWidth, ),
            columnDimensions           = ( self.termsDict[ "transformColumnDimensions" ], ),
            potentialPct               = 0.85,
            potentialRadius            = self.transformEncodingWidth,
            globalInhibition           = True,
            localAreaDensity           = 0,
            numActiveColumnsPerInhArea = self.termsDict[ "numActiveColumnsPerInhArea" ],
            synPermInactiveDec         = 0.005,
            synPermActiveInc           = 0.04,
            synPermConnected           = 0.1,
            boostStrength              = self.termsDict[ "boostStrength" ],
            seed                       = -1,
            wrapAround                 = False
        )

        self.transformClassifier = Classifier( self.termsDict[ "transformColumnDimensions" ], self.termsDict[ "numTransformDimensions" ], 10 )
        self.eyeOnSelf = True

        # Set up vector encoder parameters.
        positionEncodeParams = ScalarEncoderParameters()
        positionEncodeParams.activeBits = 5
        positionEncodeParams.radius     = 10
        positionEncodeParams.clipInput  = True
        positionEncodeParams.minimum    = -int( max( screenWidth, screenHeight ) / 2 )
        positionEncodeParams.maximum    = int( max( screenWidth, screenHeight ) / 2 )
        positionEncodeParams.periodic   = False
        self.xPosEncoder  = ScalarEncoder( positionEncodeParams )
        self.yPosEncoder  = ScalarEncoder( positionEncodeParams )
        self.positionEncodingWidth = self.xPosEncoder.size + self.yPosEncoder.size

        # The transformation input pooler.
        self.prp = SpatialPooler(
            inputDimensions            = ( self.positionEncodingWidth, ),
            columnDimensions           = ( self.termsDict[ "positionColumnDimensions" ], ),
            potentialPct               = 0.85,
            potentialRadius            = self.positionEncodingWidth,
            globalInhibition           = True,
            localAreaDensity           = 0,
            numActiveColumnsPerInhArea = self.termsDict[ "numActiveColumnsPerInhArea" ],
            synPermInactiveDec         = 0.005,
            synPermActiveInc           = 0.04,
            synPermConnected           = 0.1,
            boostStrength              = self.termsDict[ "boostStrength" ],
            seed                       = -1,
            wrapAround                 = False
        )

        self.vp = NewVectorMemory( self.termsDict )

        self.lastVector = []
        self.newVector  = []

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

    def TrainMotion( self, vector, vectorSDR ):
    # Train the classifiers on the vector componants, and the transformationSDR.

        self.transformClassifier.Learn( vectorSDR, vector )

    def InferMotion( self, objectList, sensePosX, sensePosY, vectorSDR ):
    # Infer from the vectorSDR what the movement vector is.
    # Also alternate the eye between enemy and Agent.

        newVector = []

        # Change eye position alternating from object and enemy.
        if self.eyeOnSelf:
            eyePos = objectList[ 1 ]
            self.eyeOnSelf = False
        else:
            eyePos = objectList[ 0 ]
            self.eyeOnSelf = True
        newVector.append( eyePos[ 0 ] - sensePosX )
        newVector.append( eyePos[ 1 ] - sensePosY )

        # Get list of probable inferred vectors from vectorSDR
        probableVectors = self.transformClassifier.Infer( vectorSDR )

        if len( probableVectors ) == 0:
            # Generate random motion of the agent.
            newVector.append( choice( [ -1, 1 ] ) )
            newVector.append( choice( [ -1, 1 ] ) )
        else:
            newVector.append( probableVectors[ 0 ][ 1 ][ 2 ] )
            newVector.append( probableVectors[ 0 ][ 1 ][ 3 ] )

        return newVector

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

        xEyeBits  = self.xEyeEncoder.encode( vector[ 0 ] )
        yEyeBits  = self.yEyeEncoder.encode( vector[ 1 ] )
        xMoveBits = self.xMoveEncoder.encode( vector[ 2 ] )
        yMoveBits = self.yMoveEncoder.encode( vector[ 3 ] )

        encoding     = SDR( self.transformEncodingWidth ).concatenate( [ xEyeBits, yEyeBits, xMoveBits, yMoveBits ] )
        transformSDR = SDR( self.trp.getColumnDimensions() )
        self.trp.compute( encoding, True, transformSDR )

        return transformSDR

    def EncodePositionData( self, position ):
    # Get position and encode it as an SDR.

        xPosBits  = self.xPosEncoder.encode( position[ 0 ] )
        yPosBits  = self.yPosEncoder.encode( position[ 1 ] )

        encoding     = SDR( self.positionEncodingWidth ).concatenate( [ xPosBits, yPosBits ] )
        positionSDR = SDR( self.prp.getColumnDimensions() )
        self.prp.compute( encoding, True, positionSDR )

        return positionSDR

    def Brain ( self, objectList, sensePosX, sensePosY, lastVector, lastPosition, noisePct, randomInput ):
    # The central brain function of the agent.

        self.lastVector = lastVector.copy()

        if randomInput:
            noisePct = 1.00

        # Encode the input column SDR for current position.
        senseSDR = self.EncodeSenseData( sensePosX, sensePosY, objectList, noisePct )

        # Encode the transformation vector and get column SDR.
        lastVectorSDR   = self.EncodeTransformationData( lastVector ).sparse.tolist()
        # Encode the position and get its columns SDR.
        lastPositionSDR = self.EncodePositionData( lastPosition ).sparse.tolist()

        # Train the classifiers on the vector and transformSDR
        self.TrainMotion( lastVector, lastVectorSDR )

        # Compute the action of vector memory, and learn on the synapses.
        newVectorSDR = self.vp.Compute( senseSDR, lastVectorSDR, lastPositionSDR )

        # Decode the newVectorSDR into a vector.
        newVector1 = self.InferMotion( objectList, sensePosX, sensePosY, newVectorSDR )

        self.newVector = newVector1.copy()

        return newVector1
