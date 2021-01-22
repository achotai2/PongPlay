import numpy
import turtle
import random
import time
import sys
from math import sqrt

from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
import htm.bindings.encoders
ScalarEncoder           = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters
RDSE                    = htm.bindings.encoders.RDSE
RDSE_Parameters         = htm.bindings.encoders.RDSE_Parameters
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory

screenHeight = 600          # Used in setting up screen and encoders
screenWidth = 800

# Set up turtle screen.
wn = turtle.Screen( )
wn.title( "2D" )
wn.bgcolor( "black" )
wn.setup( width=screenWidth, height=screenHeight )
wn.tracer( 0 )

# Set up agent.
agentDraw = turtle.Turtle( )
agentDraw.speed( 0 )
agentDraw.shape( "square" )
agentDraw.color( "white" )
agentDraw.shapesize( stretch_wid=2, stretch_len=2 )
agentDraw.penup( )
agentDraw.goto( 0, 0 )

# Set up den.
denDraw = turtle.Turtle( )
denDraw.speed( 0 )
denDraw.shape( "square" )
denDraw.color( "blue" )
denDraw.shapesize( stretch_wid=2, stretch_len=2 )
denDraw.penup( )
denDraw.goto( 0, 0 )

# Set up place cell stuff.
placeCellsDraw = []
for i in range( -int( screenWidth / 40 ), int( screenWidth / 40 ) ):
    currPlaceDraw = turtle.Turtle( )
    currPlaceDraw.speed( 0 )
    currPlaceDraw.shape( "circle" )
    currPlaceDraw.color( "red" )
    currPlaceDraw.shapesize( stretch_wid=0.5, stretch_len=0.5 )
    currPlaceDraw.penup( )
    currPlaceDraw.goto( i * 20, 0 )
    placeCellsDraw.append( currPlaceDraw )

# Functions
def agent_up( ):
    y = agentDraw.ycor( )
    if y < 290 - 20:
        y += 20
        agentDraw.sety(y)

def agent_down( ):
    y = agentDraw.ycor( )
    if y > -290 + 20:
        y -= 20
        agentDraw.sety(y)

def agent_right( ):
    x = agentDraw.xcor( )
    if x < 390 - 20:
        x += 20
        agentDraw.setx(x)

def agent_left( ):
    x = agentDraw.xcor( )
    if x > -390 + 20:
        x -= 20
        agentDraw.setx(x)

def Overlap ( SDR1, SDR2 ):
# Computes overlap score between two passed SDRs.

    overlap = 0

    for cell1 in SDR1.sparse:
        if cell1 in SDR2.sparse:
            overlap += 1

    return overlap

def GreatestOverlap ( testSDR, listSDR, threshold ):
# Finds SDR in listSDR with greatest overlap with testSDR and returns it, and its index in the list.
# If none are found above threshold or if list is empty it returns an empty SDR of length testSDR.

    greatest = [ 0, -1, SDR( testSDR.size ) ]

    if len( listSDR ) > 0:
        # The first element of listSDR should always be a union of all the other SDRs in list,
        # so a check can be performed first.
        if Overlap( testSDR, listSDR[0] ) >= threshold:
            aboveThreshold = []
            for idx, checkSDR in enumerate( listSDR ):
                if idx != 0:
                    thisOverlap = Overlap( testSDR, checkSDR )
                    if thisOverlap >= threshold:
                        aboveThreshold.append( [ thisOverlap, idx, checkSDR ] )
            if len( aboveThreshold ) > 0:
                greatest = sorted( aboveThreshold, key=lambda tup: tup[0], reverse=True )[ 0 ]

    return greatest

def AnyInThreshold ( testSDR, listSDR, lowThreshold, highThreshold ):
# Checks if testSDR is in listSDR with a threshold between low and highThreshold.

    if len( listSDR ) > 0:
        if Overlap( testSDR, listSDR[0] ) >= lowThreshold:
            for checkSDR in listSDR:
                thisOverlap = Overlap( testSDR, checkSDR )
                if thisOverlap > lowThreshold and thisOverlap < highThreshold:
                    return True

    return False

class VectorNet:

    def __init__( self, inputSDRsize, numColumns, maxActive, synInactiveDec, synActiveInc ):

        self.inputDim       = inputSDRsize
        self.numColumns     = numColumns
        self.maxActive      = maxActive
        self.synInactiveDec = synInactiveDec
        self.synActiveInc   = synActiveInc

        self.synapse = numpy.random.uniform( low=-0.05, high=0.05, size=( inputSDRsize * numColumns, ) )

        self.activeColumns = []
        self.ranActivate   = False

    def ActivateColumns( self, inputSDR ):
    # Feed forward cells from inputSDR, along synapses, to activate columns in vector representation.
    # Then return vector representation.

        # Feed forward inputSDR along synapses to get column activation.
        columnActivation = numpy.zeros( self.numColumns, dtype=numpy.float32 )
        for col in range( self.numColumns ):
            for cell in inputSDR.sparse:
                columnActivation[ col ] += self.synapse[ ( cell * self.numColumns ) + col ]

        # Find the top active columns, numbering maxActive, and feed the activation into these.
        maxActiveColumns = numpy.absolute( columnActivation ).argsort( )[ -self.maxActive: ][ ::-1 ]
        transformActive = numpy.zeros( self.numColumns, dtype=numpy.float32 )
        for act in maxActiveColumns:
            transformActive[ act ] = columnActivation[ act ]

        self.activeColumns = maxActiveColumns
        self.ranActivate   = True

        return transformActive

    def AdjustSynapse( self, colID, inputSDR, lessThan ):
    # Adjust the active synapse connections to reduce error between actual vector and projected vector.
    # If lessThan is False then decrease active synapses connecting to that column, if True then increase them.

        if not self.ranActivate:
            sys.exit( "Custom Error: Run ActivateColumns() before AdjustSynapse()" )

        if colID in self.activeColumns:
            inc = self.synActiveInc
        else:
            inc = self.synInactiveDec

        if lessThan:
            flip = 1
        else:
            flip = -1

        for cell in inputSDR.sparse:
            self.synapse[ ( cell * self.numColumns ) + colID ] += inc * flip

class Agent:

#    placeCellThresholdLow = 15
#    placeCellThresholdHigh = 30

    # Set up encoder parameters
    agentXEncodeParams = ScalarEncoderParameters( )
    agentYEncodeParams = ScalarEncoderParameters( )

    agentXEncodeParams.activeBits = 40
    agentXEncodeParams.radius     = 50
    agentXEncodeParams.clipInput  = False
    agentXEncodeParams.minimum    = -screenWidth / 2
    agentXEncodeParams.maximum    = screenWidth / 2
    agentXEncodeParams.periodic   = False

    agentYEncodeParams.activeBits = 40
    agentYEncodeParams.radius     = 50
    agentYEncodeParams.clipInput  = False
    agentYEncodeParams.minimum    = -screenHeight / 2
    agentYEncodeParams.maximum    = screenHeight / 2
    agentYEncodeParams.periodic   = False

    def __init__( self ):

#        self.firstRun = True
        self.timeStep = 0
#        self.placeCells = []
#        self.placeCellVect = []
#        self.lastPlaceCell = -1
        self.goalOrigin = False

        # Set up encoders
        self.agentEncoderX    = ScalarEncoder( self.agentXEncodeParams )
        self.agentEncoderY    = ScalarEncoder( self.agentYEncodeParams )

        self.encodingWidth = ( self.agentEncoderX.size + self.agentEncoderY.size )

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
            wrapAround                 = True
        )

        self.tp = TemporalMemory(
            columnDimensions          = ( 2048, ),
            cellsPerColumn            = 10,
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

        self.vectorNetwork = VectorNet( self.tp.getCellsPerColumn() * self.tp.getColumnDimensions()[ 0 ], 1, 5, 0.01, 0.05 )

        self.lastSDR = SDR( self.sp.getColumnDimensions() )
#        self.activeVector = numpy.zeros( self.vectorNetwork.numColumns, dtype=numpy.float32 )
        self.lastVect = numpy.zeros( self.vectorNetwork.numColumns, dtype=numpy.float32 )

#        self.spColumnElasticity = numpy.zeros( self.sp.getColumnDimensions()[ 0 ], dtype=numpy.float32 )
        self.spColumnConnects   = []
        for i in range( self.sp.getColumnDimensions()[ 0 ] ):
            self.spColumnConnects.append( numpy.random.uniform( low=-10, high=10, size=( self.vectorNetwork.numColumns, ) ) )

        self.motorDecoderSynapse = numpy.random.uniform( low=0.00, high=0.05, size=( self.vectorNetwork.numColumns * 2, ) )

        self.memBuffer = []

    def EncodeSenseData( self, agentX, agentY ):
    # Encodes sense data as an SDR and returns it.

        # Now we call the encoders to create bit representations for each value, and encode them.
        agentBitsX    = self.agentEncoderX.encode( agentX )
        agentBitsY    = self.agentEncoderY.encode( agentY )

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( self.encodingWidth ).concatenate( [ agentBitsX, agentBitsY ] )

        return encoding

    def ActivateColumns( self, thisPlaceSDR, lastPlaceSDR, learn ):
    # Use the inputSDR to activate cells, ticking over to next cell in active column.
    # Only activate number of cells equal to maxActive.

    # REQUIRED CHANGES:
    #   - Need to speed it up, it runs slow right now.
    #   - Each place cell stores a vector, and whenever you arrive at that vector the synapses of the vector
    #       move a bit in the direction of the arrived vector, and the synapses of the arrived vector move
    #       a bit in the direction of the place cell vector.
    # It should work. If you imagine this taken to the extreme of learning for two place cells then you could have
    # one column representing distance along x-axis and another y-axis, and transformation between SDRs would map
    # to corresponding excitation or inhibition of these columns to make the displacement vectors work.

        # Form transition SDRs by taking the intersection of the passed place cells, ie. symmetries.
        intersectSDR = SDR ( self.vectorNetwork.inputSDRsize )
        intersectSDR.sparse = numpy.intersect1d( thisPlace.sparse, lastPlace.sparse )
        # Those cells that turned off in transition.
        turnOffSDR = SDR ( self.vectorNetwork.inputSDRsize )
        turnOffSDR.sparse = numpy.setdiff1d( lastPlace.sparse, intersectSDR.sparse )
        # Those cells that turned on in transition.
        turnOnSDR = SDR ( self.vectorNetwork.inputSDRsize )
        turnOnSDR.sparse = numpy.setdiff1d( thisPlace.sparse, intersectSDR.sparse )

        # Feed forward inputSDR along synapses to get column activation.
        columnActivation = numpy.zeros( self.vectorNetwork.numColumns, dtype=numpy.float32 )
        for col in range( self.vectorNetwork.numColumns ):
            for intersectCell in intersectSDR.sparse:
                columnActivation[ col ] += self.vectorNetwork.synapse[ ( intersectCell * self.vectorNetwork.numColumns ) + col ]
            for offCell in turnOffSDR.sparse:
                columnActivation[ col ] += self.vectorNetwork.synapse[ ( ( self.vectorNetwork.inputSDRsize + offCell ) * self.vectorNetwork.numColumns ) + col ]
            for onCell in turnOnSDR.sparse:
                columnActivation[ col ] += self.vectorNetwork.synapse[ ( ( ( self.vectorNetwork.inputSDRsize * 2 ) + onCell ) * self.vectorNetwork.numColumns ) + col ]

        # Find the top active columns, numbering maxActive, and feed the activation into these.
        maxActiveColumns = numpy.absolute( columnActivation ).argsort( )[ -self.vectorNetwork.maxActive: ][ ::-1 ]
        transformActive = numpy.zeros( self.vectorNetwork.numColumns, dtype=numpy.float32 )
        for act in maxActiveColumns:
            transformActive[ act ] = columnActivation[ act ]

        # If this is the first time seeing this location, store vector for this place by adding transformation
        # activation to lastPlace vector.
#        summedVector = numpy.add( transformActive, self.placeCellVect[ lastPlace ] )
#        if firstRun:
#            self.placeCellVect[ thisPlace ] = summedVector

        # Perform learning on synapses by testing vector space arithmetic.
        if learn:
            for col in range( self.vectorNetwork.numColumns ):

#                if summedVector[ col ] >= self.placeCellVect[ thisPlace ][ col ]:
#                    # Adjust lastPlace vector and thisPlace vector to reduce error in summedVector.
#                    if thisPlace != 1:          # Don't alter the vector for den element, it is the origin.
#                        self.placeCellVect[ thisPlace ][ col ] += self.vectorNetwork.synActiveInc
#                    if lastPlace != 1:
#                        self.placeCellVect[ lastPlace ][ col ] -= self.vectorNetwork.synActiveInc

                    # Adjust synapse connections (which produces transformation vector) to reduce error in summedVector.
                    for intersectCell in intersectSDR.sparse:
                        self.vectorNetwork.synapse[ ( intersectCell * self.vectorNetwork.numColumns ) + col ] -= self.vectorNetwork.synInactiveDec
                    for offCell in intersectSDR.sparse:
                        self.vectorNetwork.synapse[ ( ( self.vectorNetwork.inputSDRsize + offCell ) * self.vectorNetwork.numColumns ) + col ] -= self.vectorNetwork.synInactiveDec
                    for onCell in intersectSDR.sparse:
                        self.vectorNetwork.synapse[ ( ( ( self.vectorNetwork.inputSDRsize * 2 ) + onCell ) * self.vectorNetwork.numColumns ) + col ] -= self.vectorNetwork.synInactiveDec
#                else:
                    # Adjust lastPlace vector and thisPlace vector to reduce error in summedVector.
#                    if thisPlace != 1:
#                        self.placeCellVect[ thisPlace ][ col ] -= self.vectorNetwork.synActiveInc
#                    if thisPlace != 1:
#                        self.placeCellVect[ lastPlace ][ col ] += self.vectorNetwork.synActiveInc

                    # Adjust synapse connections (which produces transformation vector) to reduce error in summedVector.
                    for intersectCell in intersectSDR.sparse:
                        self.vectorNetwork.synapse[ ( intersectCell * self.vectorNetwork.numColumns ) + col ] += self.vectorNetwork.synInactiveDec
                    for offCell in intersectSDR.sparse:
                        self.vectorNetwork.synapse[ ( ( self.vectorNetwork.inputSDRsize + offCell ) * self.vectorNetwork.numColumns ) + col ] += self.vectorNetwork.synInactiveDec
                    for onCell in intersectSDR.sparse:
                        self.vectorNetwork.synapse[ ( ( ( self.vectorNetwork.inputSDRsize * 2 ) + onCell ) * self.vectorNetwork.numColumns ) + col ] += self.vectorNetwork.synInactiveDec

# Keyboard bindings
wn.listen()
wn.onkey( agent_up, "w" )
wn.onkey( agent_down, "s" )
wn.onkey( agent_left, "a" )
wn.onkey( agent_right, "d" )

agentClass = Agent()

while True:
# Main game loop

    wn.update( )         # Screen update

    agentClass.timeStep += 1

    if agentClass.timeStep >= 50:
        print ("Go Home!-------------------------------------------------------------------------")
        agentClass.goalOrigin = True
    if agentDraw.xcor() == 0 and agentDraw.ycor() == 0:
        agentClass.timeStep = 0
        agentClass.vectorNetwork.activeCells = numpy.zeros( agentClass.vectorNetwork.numColumns, dtype=numpy.float32 )
        agentClass.goalOrigin = False
#        agentClass.memBuffer.clear()

    # Encode sense data.
    encoding = agentClass.EncodeSenseData( agentDraw.xcor(), agentDraw.ycor() )

    # Get place cell SDR through spatial pooler.
    senseSDR = SDR( agentClass.sp.getColumnDimensions() )
    agentClass.sp.compute( encoding, False, senseSDR )

    # Generate current place cell vector.
    senseVector = numpy.zeros( agentClass.vectorNetwork.numColumns, dtype=numpy.float32 )
    for col in range( agentClass.vectorNetwork.numColumns ):
        columnValue = 0.0
        for active in senseSDR.sparse:
            columnValue += agentClass.spColumnConnects[ active ][ col ]
        if agentDraw.xcor() != 0 or agentDraw.ycor() != 0:
            senseVector[ col ] = columnValue
    placeCellsDraw[ int( ( agentDraw.xcor() / 20 ) + ( screenWidth / 40 ) ) ].sety( senseVector[ 0 ] )
    # If we are at origin then train vector to be zero.
    if agentDraw.xcor() == 0 and agentDraw.ycor() == 0:
        for active in senseSDR.sparse:
            for col in range( agentClass.vectorNetwork.numColumns ):
                if senseVector[ col ] > 0:
                    agentClass.spColumnConnects[ active ][ col ] -= 0.001
                elif senseVector[ col ] < 0:
                    agentClass.spColumnConnects[ active ][ col ] += 0.001

    # Feed the last place SDR and then this one through the temporal predictor to get the vectorSDR.
    agentClass.tp.reset()
    agentClass.tp.compute( agentClass.lastSDR, True )
    agentClass.tp.compute( senseSDR, True )
    transVectorSDR = agentClass.tp.getActiveCells()

    # Get vector representation of transformation from transVectorSDR.
    transVectorRep = agentClass.vectorNetwork.ActivateColumns( transVectorSDR )

#    # Add transformation vector to active vector.
#    agentClass.activeVector = numpy.add( agentClass.activeVector, transVectorRep )

    # Add transformation vector to old vector.
    predVectorRep = numpy.add( agentClass.lastVect, transVectorRep )

    # Support elasticity of encountered cells based on what we last saw.
#    if agentDraw.xcor() == 0 and agentDraw.ycor() == 0:
#        for cell in senseSDR.sparse:
#            agentClass.spColumnElasticity[ cell ] += 1
#        agentClass.activeVector = numpy.zeros( agentClass.vectorNetwork.numColumns, dtype=numpy.float32 )
#    else:
#        sum = 0
#        for cell in agentClass.lastSDR.sparse:
#            sum += agentClass.spColumnElasticity[ cell ]
#        for cell in senseSDR.sparse:
#            agentClass.spColumnElasticity[ cell ] += 0.1 * ( 1 - numpy.exp( -sum / 100 ) )

#    for col in range( agentClass.vectorNetwork.numColumns ):
#        # Generate current place cell vector.
#        columnValue = 0
#        for active in senseSDR.sparse:
#            columnValue += agentClass.spColumnConnects[ active ][ col ]
#
#        print ( "----------------------------" )
#        print ( "Elasticity:", agentClass.spColumnElasticity[ col ] )
#        print ( "ColumnValue:", columnValue )
#        print ( "activeVector:", agentClass.activeVector[ col ] )
#
#        if columnValue >= transVectorRep[ col ]:
#            columnValue = 0                                             # THIS IS FOR THE PRINT
#            # Change spColumnConnects connections to make place vector closer to predVectorRep.
#            for active in senseSDR.sparse:
#                agentClass.spColumnConnects[ active ][ col ] -= 0.01 )
#                columnValue += agentClass.spColumnConnects[ active ][ col ]         # THIS IS FOR THE PRINT
#
#            # Change vectorNetwork synapses to make predVectorRep closer to  place vector.
#            agentClass.vectorNetwork.AdjustSynapse( col, transVectorSDR, lessThan=True )
#        else:
#            columnValue = 0                                 # THIS IS FOR THE PRINT
#            # Change spColumnConnects connections to make place vector closer to predVectorRep.
#            for active in senseSDR.sparse:
#                agentClass.spColumnConnects[ active][ col ] += 0.01 )
#                columnValue += agentClass.spColumnConnects[ active ][ col ]         # THIS IS FOR THE PRINT

            # Change vectorNetwork synapses to make predVectorRep closer to  place vector.
#            agentClass.vectorNetwork.AdjustSynapse( col, transVectorSDR, lessThan=False )

#        print( "Adjusted ColumnValue:", columnValue )

#    agentClass.vectorNetwork.ranActivate = False

    if len( agentClass.memBuffer ) > 0:
        # Get the cells that turned on from lastSDR to senseSDR. These are the only ones we will check for cycles.
        intersectSDR = numpy.intersect1d( senseSDR.sparse, agentClass.memBuffer[ 0 ][ 0 ].sparse, assume_unique=True )
        turnOnSDR    = numpy.setdiff1d( senseSDR.sparse, intersectSDR, assume_unique=True )

        # Look for the last occurrence of this cell being on.
        for on in turnOnSDR:
            cycleOrigin = -1
            for buffIdx in range( len( agentClass.memBuffer ) ):
                if on in agentClass.memBuffer[ buffIdx ][ 0 ].sparse:
                    cycleOrigin = buffIdx
                    break

            # If we found one then perform learning on it.
            if cycleOrigin != -1:
                # Assume simple linear motion away and then back.
                cWidth  = ( cycleOrigin + 1 ) / 2
                cCentre = numpy.ceil( cWidth )
                cFar    = agentClass.memBuffer[ int( cCentre ) ][ 1 ][ 0 ]          # FOR NOW WE TAKE JUST 0 DIMENSION
                cOrigin = agentClass.memBuffer[ cycleOrigin ][ 1 ][ 0 ]             # FOR NOW WE TAKE JUST 0 DIMENSION

                # Calculate what the cycle thinks this dimension should be for the elements in the cycle.
                for learnIdx in range( buffIdx ):
                    if cFar >= cOrigin:
                        predValue = -( ( cFar - cOrigin ) / cWidth ) * numpy.absolute( ( learnIdx - cCentre ) ) + cFar
                    else:
                        predValue = ( ( cOrigin - cFar ) / cWidth ) * numpy.absolute( ( learnIdx - cCentre ) ) + cFar

                    if agentClass.memBuffer[ learnIdx ][ 1 ][ 0 ] > predValue:
                        for active in agentClass.memBuffer[ learnIdx ][ 0 ].sparse:
                            for col in range( agentClass.vectorNetwork.numColumns ):
                                agentClass.spColumnConnects[ active ][ col ] -= 0.005
                    elif agentClass.memBuffer[ learnIdx ][ 1 ][ 0 ] < predValue:
                        for active in agentClass.memBuffer[ learnIdx ][ 0 ].sparse:
                            for col in range( agentClass.vectorNetwork.numColumns ):
                                agentClass.spColumnConnects[ active ][ col ] += 0.005

    # Add stuff to buffer.
    agentClass.memBuffer.insert( 0, [ senseSDR, senseVector ] )
    if len( agentClass.memBuffer ) >= 100:
        agentClass.memBuffer.pop( -1 )

    # If we arrive at the den then learn on the sequence.
#    if agentDraw.xcor() == 0 and agentDraw.ycor() == 0:
        # Calculate difference between found zero vector and actual zero vector.
        # Since zero vector is all zeros this is just the agentClass.activeVector.

        # Go through inputBuffer and adjust weights of transformation according to difference.

        # Use

    # Check senseSDR against stored place cells.
#    if not agentClass.firstRun:
#        greatestOverlap = GreatestOverlap( senseSDR, agentClass.placeCells, agentClass.placeCellThresholdLow )

        # Set up origin point as den, and make sure the agent always know it is at the den.
#        if agentDraw.xcor() == 0 and agentDraw.ycor() == 0:
#            # Add in den place cell.
#            if len( agentClass.placeCells ) == 0:
#                agentClass.placeCells.append( SDR( senseSDR.size ) )           # Union element.
#                agentClass.placeCellVect.append( numpy.zeros( agentClass.vectorNetwork.numColumns, dtype=numpy.float32 ) )
#                agentClass.placeCells.append( senseSDR )                       # Den element
#                agentClass.placeCellVect.append( numpy.zeros( agentClass.vectorNetwork.numColumns, dtype=numpy.float32 ) )
#                agentClass.placeCells[0].sparse = numpy.union1d( agentClass.placeCells[ 0 ].sparse, senseSDR.sparse )

#            if greatestOverlap[ 1 ] != 1:
#                # Delete place cell mistakingly identified as den.
#                if greatestOverlap [ 1 ] != -1:
#                    if agentClass.lastPlaceCell > greatestOverlap[ 1 ]:
#                        agentClass.lastPlaceCell -= 1
#                    elif agentClass.lastPlaceCell == greatestOverlap[ 1 ]:
#                        agentClass.lastPlaceCell = 1
#                    agentClass.placeCells.pop( greatestOverlap[ 1 ] )
#                greatestOverlap[ 1 ] = 1
#                greatestOverlap[ 0 ] = agentClass.placeCellThresholdHigh

#        if greatestOverlap[ 1 ] == -1:
#            # If it is lower than placeCellThresholdLow insert and draw a new place cell.
#            placeDraw = turtle.Turtle( )
#            placeDraw.color( "red" )
#            placeDraw.penup( )
#            placeDraw.setposition( agentDraw.xcor( ), agentDraw.ycor( ) - 20 )
#            placeDraw.pendown( )
#            placeDraw.circle( 20 )
#            placeCellsDraw.append( placeDraw )
#            currPlaceDraw.setx( agentDraw.xcor( ) )
#            currPlaceDraw.sety( agentDraw.ycor( ) )

            # Add new place cell and union with first element.
#            agentClass.placeCells.append( senseSDR )
#            agentClass.placeCellVect.append( numpy.zeros( agentClass.vectorNetwork.numColumns, dtype=numpy.float32 ) )
#            agentClass.placeCells[0].sparse = numpy.union1d( agentClass.placeCells[ 0 ].sparse, senseSDR.sparse )

            # Call ActivateColumns and update lastPlaceCell
#            if agentClass.lastPlaceCell != -1:
#                agentClass.ActivateColumns( -1, agentClass.lastPlaceCell, True, True )
#            agentClass.lastPlaceCell = len( agentClass.placeCells ) - 1

#        elif greatestOverlap[ 0 ] >= agentClass.placeCellThresholdHigh:
#            # If it is higher than placeCellThresholdHigh then we are on a place cell.
#            currPlaceDraw.setx( placeCellsDraw[ greatestOverlap[ 1 ] - 1 ].xcor( ) )
#            currPlaceDraw.sety( placeCellsDraw[ greatestOverlap[ 1 ] - 1 ].ycor( ) )

#            if agentClass.lastPlaceCell != -1:
#                agentClass.ActivateColumns( greatestOverlap[ 1 ], agentClass.lastPlaceCell, True, False )
#            agentClass.lastPlaceCell = greatestOverlap[ 1 ]

    # Move back to the origin when triggered to.
    if agentClass.goalOrigin:
        # Try to move towards origin.
#        whichWay = numpy.zeros( 2, dtype=numpy.float32 )
#        for idx in range( agentClass.vectorNetwork.numColumns ):
#            whichWay[ 0 ] -= agentClass.motorDecoderSynapse[ idx ] * agentClass.activeVector[ idx ]
#            whichWay[ 1 ] -= agentClass.motorDecoderSynapse[ idx + agentClass.vectorNetwork.numColumns ] * agentClass.activeVector[ idx ]

#        if whichWay[ 0 ] >= whichWay[ 1 ]:
#            motorOutput = 0
#        else:
#            motorOutput = 1
        if agentDraw.xcor() >= 0:
            motorOutput = 0
        else:
            motorOutput = 1

    else:
        # Randomly choose motor output.
        motorOutput = random.randint( 0, 1 )
#    if motorOutput == 0:
#        agent_up()
#    elif motorOutput == 1:
#        agent_down()
    if motorOutput == 0:
        agent_left()
    elif motorOutput == 1:
        agent_right()

    # Train motor decoder using chosen motor action and transformation vector by finding what it thinks is the
    # chosen motor action, and then supporting or inhibiting the synapses if it is right or wrong.
    for idx in range( agentClass.vectorNetwork.numColumns ):
        if transVectorRep[ idx ] != 0:
            if motorOutput == 0:
                agentClass.motorDecoderSynapse[ idx ]                                       += 0.01
                agentClass.motorDecoderSynapse[ idx + agentClass.vectorNetwork.numColumns ] -= 0.01
            else:
                agentClass.motorDecoderSynapse[ idx ]                                       -= 0.01
                agentClass.motorDecoderSynapse[ idx + agentClass.vectorNetwork.numColumns ] += 0.01

    # Change the lastSDR to this one for next time step.
    agentClass.lastSDR = senseSDR
