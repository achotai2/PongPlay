import numpy
import turtle
import random
import time

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

# Set up place cell stuff.
placeCellsDraw = []
currPlaceDraw = turtle.Turtle( )
currPlaceDraw.speed( 0 )
currPlaceDraw.shape( "circle" )
currPlaceDraw.color( "red" )
currPlaceDraw.shapesize( stretch_wid=0.5, stretch_len=0.5 )
currPlaceDraw.penup( )
currPlaceDraw.goto( 0, 0 )

# Set up den.
denDraw = turtle.Turtle( )
denDraw.speed( 0 )
denDraw.shape( "square" )
denDraw.color( "blue" )
denDraw.shapesize( stretch_wid=2, stretch_len=2 )
denDraw.penup( )
denDraw.goto( 0, 0 )

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

class CyclicNet:

    def __init__( self, inputDim, numColumns, maxActive, synInactiveDec, synActiveInc ):

        self.inputDim       = inputDim
        self.numColumns     = numColumns
        self.maxActive      = maxActive
        self.synInactiveDec = synInactiveDec
        self.synActiveInc   = synActiveInc

        self.synapse     = numpy.random.uniform( low=-0.05, high=0.05, size=( inputDim * numColumns, ) )

class Agent:

    placeCellThresholdLow = 15
    placeCellThresholdHigh = 30

    # Set up encoder parameters
    agentXEncodeParams = ScalarEncoderParameters( )
    agentYEncodeParams = ScalarEncoderParameters( )

    agentXEncodeParams.activeBits = 51
    agentXEncodeParams.radius     = 50
    agentXEncodeParams.clipInput  = False
    agentXEncodeParams.minimum    = -screenWidth / 2
    agentXEncodeParams.maximum    = screenWidth / 2
    agentXEncodeParams.periodic   = False

    agentYEncodeParams.activeBits = 51
    agentYEncodeParams.radius     = 50
    agentYEncodeParams.clipInput  = False
    agentYEncodeParams.minimum    = -screenHeight / 2
    agentYEncodeParams.maximum    = screenHeight / 2
    agentYEncodeParams.periodic   = False

    def __init__( self ):

        self.firstRun = True
        self.timeStep = 0
        self.placeCells = []
        self.placeCellVect = []
        self.lastPlaceCell = -1

        # Set up encoders
        self.agentEncoderX    = ScalarEncoder( self.agentXEncodeParams )
        self.agentEncoderY    = ScalarEncoder( self.agentYEncodeParams )

        self.encodingWidth = ( self.agentEncoderX.size + self.agentEncoderY.size )

        self.sp = SpatialPooler(
            inputDimensions            = (self.encodingWidth,),
            columnDimensions           = (2048,),
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

        self.cycNetwork = CyclicNet( self.sp.getColumnDimensions( )[ 0 ] * 3, 100, 5, 0.05, 0.1 )

    def EncodeSenseData( self, agentX, agentY ):
    # Encodes sense data as an SDR and returns it.

        # Now we call the encoders to create bit representations for each value, and encode them.
        agentBitsX    = self.agentEncoderX.encode( agentX )
        agentBitsY    = self.agentEncoderY.encode( agentY )

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( self.encodingWidth ).concatenate( [ agentBitsX, agentBitsY ] )
        senseSDR = SDR( self.sp.getColumnDimensions( ) )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    def ActivateColumns( self, thisPlace, lastPlace, learn, firstRun ):
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

        SDRsize = self.sp.getColumnDimensions( )[ 0 ]

        # Form transition SDRs by taking the intersection of the passed place cells, ie. symmetries.
        intersectSDR = SDR ( SDRsize )
        intersectSDR.sparse = numpy.intersect1d( self.placeCells[ thisPlace ].sparse, self.placeCells[ lastPlace ].sparse )
        # Those cells that turned off in transition.
        turnOffSDR = SDR ( SDRsize )
        turnOffSDR.sparse = numpy.setdiff1d( self.placeCells[ lastPlace ].sparse, intersectSDR.sparse )
        # Those cells that turned on in transition.
        turnOnSDR = SDR ( SDRsize )
        turnOnSDR.sparse = numpy.setdiff1d( self.placeCells[ thisPlace ].sparse, intersectSDR.sparse )

        # Feed forward inputSDR along synapses to get column activation.
        columnActivation = numpy.zeros( self.cycNetwork.numColumns, dtype=numpy.float32 )
        for col in range( self.cycNetwork.numColumns ):
            for intersectCell in intersectSDR.sparse:
                columnActivation[ col ] += self.cycNetwork.synapse[ ( intersectCell * self.cycNetwork.numColumns ) + col ]
            for offCell in turnOffSDR.sparse:
                columnActivation[ col ] += self.cycNetwork.synapse[ ( ( SDRsize + offCell ) * self.cycNetwork.numColumns ) + col ]
            for onCell in turnOnSDR.sparse:
                columnActivation[ col ] += self.cycNetwork.synapse[ ( ( ( SDRsize * 2 ) + onCell ) * self.cycNetwork.numColumns ) + col ]

        # Find the top active columns, numbering maxActive, and feed the activation into these.
        maxActiveColumns = numpy.absolute( columnActivation ).argsort( )[ -self.cycNetwork.maxActive: ][ ::-1 ]
        transformActive = numpy.zeros( self.cycNetwork.numColumns, dtype=numpy.float32 )
        for act in maxActiveColumns:
            transformActive[ act ] = columnActivation[ act ]

        # If this is the first time seeing this location, store vector for this place by adding transformation
        # activation to lastPlace vector.
        summedVector = numpy.add( transformActive, self.placeCellVect[ lastPlace ] )
        if firstRun:
            self.placeCellVect[ thisPlace ] = summedVector

        # Perform learning on synapses by testing vector space arithmetic.
        if learn:
            for col in range( self.cycNetwork.numColumns ):

                if summedVector[ col ] >= self.placeCellVect[ thisPlace ][ col ]:
                    # Adjust lastPlace vector and thisPlace vector to reduce error in summedVector.
                    if thisPlace != 1:          # Don't alter the vector for den element, it is the origin.
                        self.placeCellVect[ thisPlace ][ col ] += self.cycNetwork.synActiveInc
                    if lastPlace != 1:
                        self.placeCellVect[ lastPlace ][ col ] -= self.cycNetwork.synActiveInc

                    # Adjust synapse connections (which produces transformation vector) to reduce error in summedVector.
                    for intersectCell in intersectSDR.sparse:
                        self.cycNetwork.synapse[ ( intersectCell * self.cycNetwork.numColumns ) + col ] -= self.cycNetwork.synInactiveDec
                    for offCell in intersectSDR.sparse:
                        self.cycNetwork.synapse[ ( ( SDRsize + offCell ) * self.cycNetwork.numColumns ) + col ] -= self.cycNetwork.synInactiveDec
                    for onCell in intersectSDR.sparse:
                        self.cycNetwork.synapse[ ( ( ( SDRsize * 2 ) + onCell ) * self.cycNetwork.numColumns ) + col ] -= self.cycNetwork.synInactiveDec
                else:
                    # Adjust lastPlace vector and thisPlace vector to reduce error in summedVector.
                    if thisPlace != 1:
                        self.placeCellVect[ thisPlace ][ col ] -= self.cycNetwork.synActiveInc
                    if thisPlace != 1:
                        self.placeCellVect[ lastPlace ][ col ] += self.cycNetwork.synActiveInc

                    # Adjust synapse connections (which produces transformation vector) to reduce error in summedVector.
                    for intersectCell in intersectSDR.sparse:
                        self.cycNetwork.synapse[ ( intersectCell * self.cycNetwork.numColumns ) + col ] += self.cycNetwork.synInactiveDec
                    for offCell in intersectSDR.sparse:
                        self.cycNetwork.synapse[ ( ( SDRsize + offCell ) * self.cycNetwork.numColumns ) + col ] += self.cycNetwork.synInactiveDec
                    for onCell in intersectSDR.sparse:
                        self.cycNetwork.synapse[ ( ( ( SDRsize * 2 ) + onCell ) * self.cycNetwork.numColumns ) + col ] += self.cycNetwork.synInactiveDec

# Keyboard bindings
wn.listen( )
wn.onkey(agent_up, "w")
wn.onkey(agent_down, "s")
wn.onkey(agent_left, "a")
wn.onkey(agent_right, "d")

agentClass = Agent( )

while True:
# Main game loop

    wn.update( )         # Screen update

    agentClass.timeStep += 1

    if agentClass.timeStep >= 50:
        agentClass.timeStep = 0
        agentClass.firstRun = False
        agentClass.cycNetwork.activeCells = numpy.zeros( agentClass.cycNetwork.numColumns, dtype=numpy.float32 )
        agentDraw.setx( 0 )
        agentDraw.sety( 0 )

    # Encode sense data.
    senseSDR = agentClass.EncodeSenseData( agentDraw.xcor( ), agentDraw.ycor( ) )

    # Check senseSDR against stored place cells.
    if not agentClass.firstRun:
        greatestOverlap = GreatestOverlap( senseSDR, agentClass.placeCells, agentClass.placeCellThresholdLow )

        # Set up origin point as den, and make sure the agent always know it is at the den.
        if agentDraw.xcor() == 0 and agentDraw.ycor() == 0:
            # Add in den place cell.
            if len( agentClass.placeCells ) == 0:
                agentClass.placeCells.append( SDR( senseSDR.size ) )           # Union element.
                agentClass.placeCellVect.append( numpy.zeros( agentClass.cycNetwork.numColumns, dtype=numpy.float32 ) )
                agentClass.placeCells.append( senseSDR )                       # Den element
                agentClass.placeCellVect.append( numpy.zeros( agentClass.cycNetwork.numColumns, dtype=numpy.float32 ) )
                agentClass.placeCells[0].sparse = numpy.union1d( agentClass.placeCells[ 0 ].sparse, senseSDR.sparse )

            if greatestOverlap[ 1 ] != 1:
                # Delete place cell mistakingly identified as den.
                if greatestOverlap [ 1 ] != -1:
                    if agentClass.lastPlaceCell > greatestOverlap[ 1 ]:
                        agentClass.lastPlaceCell -= 1
                    elif agentClass.lastPlaceCell == greatestOverlap[ 1 ]:
                        agentClass.lastPlaceCell = 1
                    agentClass.placeCells.pop( greatestOverlap[ 1 ] )
                greatestOverlap[ 1 ] = 1
                greatestOverlap[ 0 ] = agentClass.placeCellThresholdHigh

        if greatestOverlap[ 1 ] == -1:
            # If it is lower than placeCellThresholdLow insert and draw a new place cell.
            placeDraw = turtle.Turtle( )
            placeDraw.color( "red" )
            placeDraw.penup( )
            placeDraw.setposition( agentDraw.xcor( ), agentDraw.ycor( ) - 20 )
            placeDraw.pendown( )
            placeDraw.circle( 20 )
            placeCellsDraw.append( placeDraw )
            currPlaceDraw.setx( agentDraw.xcor( ) )
            currPlaceDraw.sety( agentDraw.ycor( ) )

            # Add new place cell and union with first element.
            agentClass.placeCells.append( senseSDR )
            agentClass.placeCellVect.append( numpy.zeros( agentClass.cycNetwork.numColumns, dtype=numpy.float32 ) )
            agentClass.placeCells[0].sparse = numpy.union1d( agentClass.placeCells[ 0 ].sparse, senseSDR.sparse )

            # Call ActivateColumns and update lastPlaceCell
            if agentClass.lastPlaceCell != -1:
                agentClass.ActivateColumns( -1, agentClass.lastPlaceCell, True, True )
            agentClass.lastPlaceCell = len( agentClass.placeCells ) - 1

        elif greatestOverlap[ 0 ] >= agentClass.placeCellThresholdHigh:
            # If it is higher than placeCellThresholdHigh then we are on a place cell.
            currPlaceDraw.setx( placeCellsDraw[ greatestOverlap[ 1 ] - 1 ].xcor( ) )
            currPlaceDraw.sety( placeCellsDraw[ greatestOverlap[ 1 ] - 1 ].ycor( ) )

            if agentClass.lastPlaceCell != -1:
                agentClass.ActivateColumns( greatestOverlap[ 1 ], agentClass.lastPlaceCell, True, False )
            agentClass.lastPlaceCell = greatestOverlap[ 1 ]

            print ( agentClass.placeCellVect[ greatestOverlap[ 1 ] ])

    # Randomly choose motor output
    motorOutput = random.randint( 0, 3 )
    if motorOutput == 0:
        agent_up( )
    elif motorOutput == 1:
        agent_down( )
    if motorOutput == 0 or 2:
        agent_left( )
    elif motorOutput == 1 or 3:
        agent_right( )
