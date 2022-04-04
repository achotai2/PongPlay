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
        self.lastColour = 1

        # Stats for end report.
        self.top_left     = []
        self.top_right    = []
        self.bottom_left  = []
        self.bottom_right = []

        self.localBitRep  = []
        self.centerX      = 0
        self.centerY      = 0

    def GetLogData( self ):
    # Get the local log data and return it.

        log_data = []

        self.PrintBitRep( log_data )

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
#        colourBits = self.colourEncoder.encode( colour )
        objBits = self.BuildLocalBitRep( sensePosX, sensePosY, objX, objY, objW, objH, objC, noisePct )

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
#        encoding = SDR( self.encodingWidth ).concatenate( [ colourBits ] )
        encoding = objBits
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

#        print( colour, self.whatColour.infer( senseSDR ) )
#        self.whatColour.learn( pattern = senseSDR, classification = colour )
        return senseSDR

    def Brain ( self, objX, objY, objW, objH, objC, sensePosX, sensePosY, noisePct ):
    # The central brain function of the agent.

        # Encode the input column SDR for current position.
        senseSDR = self.EncodeSenseData( sensePosX, sensePosY, objX, objY, objW, objH, objC, noisePct )

        # Generate random motion vector for next time step.
        self.lastVector = self.newVector.copy()

        # Compute cell activation and generate next predicted cells.
        if self.lastVector[ 2 ] == 0:
            feelingState = 0.0
        else:
            feelingState = 1.0
        if feelingState > 0.0:
            self.vp.ResetWorkingMemory()

        vpDesiredNewVector = self.vp.Compute( senseSDR, self.lastVector )

        # Generate random motion of the sense organ.
        if vpDesiredNewVector == None:
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

        elif len( vpDesiredNewVector ) != len( self.newVector ):
            print( "VP outputting vector of wrong dimensions." )
            exit()
        else:
            for i in range( len( self.newVector ) ):
                self.newVector[ i ] = vpDesiredNewVector[ i ]

        # Use FSegments to predict next set of inputs (put the cells into predictive states), given newVector.
        self.vp.PredictFCells( self.newVector )

        return self.newVector
