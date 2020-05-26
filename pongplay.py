# Simple Pong game combined with Nupic AI, in Python 2.7
# By Anand Chotai

import csv
import datetime
import numpy
import yaml

from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
import htm.bindings.encoders
ScalarEncoder = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters
RDSE = htm.bindings.encoders.RDSE
RDSE_Parameters = htm.bindings.encoders.RDSE_Parameters
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory

# Declare global variables
screenHeight = 600          # Used in setting up screen and encoders
screenWidth = 800

score_a = 0                 # Used to keep track of score
score_b = 0

xSpeed = 4                   # Speed for ball
ySpeed = 4

pastFeelingSize = 30        # Size of list of past SDRs to adjust feeling retroactively.
negativeFeeling = False     # Did something trigger negative/positive feeling response?
positiveFeeling = False
feelingAdjust = 100         # Maximum amount to adjust feeling by.

# Set up Turtle screen
import turtle

wn = turtle.Screen()
wn.title("Pong")
wn.bgcolor("black")
wn.setup(width=screenWidth, height=screenHeight)
wn.tracer(0)

# Paddle A
paddle_a = turtle.Turtle()
paddle_a.speed(0)
paddle_a.shape("square")
paddle_a.color("white")
paddle_a.shapesize(stretch_wid=5,stretch_len=1)
paddle_a.penup()
paddle_a.goto(-350, 0)

# Paddle B
paddle_b = turtle.Turtle()
paddle_b.speed(0)
paddle_b.shape("square")
paddle_b.color("white")
paddle_b.shapesize(stretch_wid=5,stretch_len=1)
paddle_b.penup()
paddle_b.goto(350, 0)

# Ball
ball = turtle.Turtle()
ball.speed(0)
ball.shape("square")
ball.color("white")
ball.penup()
ball.goto(0, 0)
ball.dx = xSpeed
ball.dy = ySpeed

# Pen to display score
pen = turtle.Turtle()
pen.speed(0)
pen.shape("square")
pen.color("white")
pen.penup()
pen.hideturtle()
pen.goto(0, 260)
pen.write("Player A: 0  Player B: 0", align="center", font=("Courier", 24, "normal"))

# Functions
def paddle_a_up():
    y = paddle_a.ycor()
    if y < 290 - 100:
        y += 20
        paddle_a.sety(y)

def paddle_a_down():
    y = paddle_a.ycor()
    if y > -290 + 100:
        y -= 20
        paddle_a.sety(y)

def paddle_b_up():
    y = paddle_b.ycor()
    if y < 290 - 100:
        y += 20
        paddle_b.sety(y)

def paddle_b_down():
    y = paddle_b.ycor()
    if y > -290 + 100:
        y -= 20
        paddle_b.sety(y)

# The main Nupic learning algorithm
# Set up encoder parameters
paddleEncodeParams    = ScalarEncoderParameters()
ballXEncodeParams     = ScalarEncoderParameters()
ballYEncodeParams     = ScalarEncoderParameters()
ballVelEncodeParams  = RDSE_Parameters()

paddleEncodeParams.activeBits = 99
paddleEncodeParams.radius     = 100
paddleEncodeParams.clipInput  = False
paddleEncodeParams.minimum    = -screenHeight / 2
paddleEncodeParams.maximum    = screenHeight / 2
paddleEncodeParams.periodic   = False

ballXEncodeParams.activeBits = 21
ballXEncodeParams.radius     = 20
ballXEncodeParams.clipInput  = False
ballXEncodeParams.minimum    = -screenWidth / 2
ballXEncodeParams.maximum    = screenWidth / 2
ballXEncodeParams.periodic   = False

ballYEncodeParams.activeBits = 21
ballYEncodeParams.radius     = 20
ballYEncodeParams.clipInput  = False
ballYEncodeParams.minimum    = -screenHeight / 2
ballYEncodeParams.maximum    = screenHeight / 2
ballYEncodeParams.periodic   = False

ballVelEncodeParams.size       = 400
ballVelEncodeParams.activeBits = 20
ballVelEncodeParams.radius     = 20

# Set up encoders
paddleAEncoder = ScalarEncoder( paddleEncodeParams )
paddleBEncoder = ScalarEncoder( paddleEncodeParams )
ballEncoderX = ScalarEncoder( ballXEncodeParams )
ballEncoderY = ScalarEncoder( ballYEncodeParams )
ballEncoderVelX = RDSE( ballVelEncodeParams )
ballEncoderVelY = RDSE( ballVelEncodeParams )

encodingWidth = (paddleAEncoder.size + paddleBEncoder.size + ballEncoderX.size + ballEncoderY.size + ballEncoderVelX.size + ballEncoderVelY.size)

# Set up Spatial Pooler and Temporal Memory
sp = SpatialPooler(
    inputDimensions            = (encodingWidth,),
    columnDimensions           = (3699,),
    potentialPct               = 0.85,
    potentialRadius            = encodingWidth,
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

tm = TemporalMemory(
    columnDimensions          = (3699,),
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
    seed=42
)

mn =  SpatialPooler(
    inputDimensions            = (2560,),
    columnDimensions           = (2562,),
    potentialPct               = 0.85,
    potentialRadius            = encodingWidth,
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

# Set up arrays for feeling
pastActive = []                     # Array of previously active SDRs
cellFeeling = [0.0] * 3699 * 32     # Array to store cell feeling state

# Keyboard bindings
wn.listen()
wn.onkey(paddle_a_up, "w")
wn.onkey(paddle_a_down, "s")
wn.onkey(paddle_b_up, "Up")
wn.onkey(paddle_b_down, "Down")

# Main game loop
while True:
    wn.update()         # Screen update

    # Move the ball
    ball.setx(ball.xcor() + ball.dx)
    ball.sety(ball.ycor() + ball.dy)

    # Border checking
    # Top and bottom
    if ball.ycor() > 290:
        ball.sety(290)
        ball.dy *= -1

    elif ball.ycor() < -290:
        ball.sety(-290)
        ball.dy *= -1

    # Left and right
    if ball.xcor() > 350:
        score_a += 1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a, score_b), align="center", font=("Courier", 24, "normal"))
        ball.goto(0, 0)
        ball.dx *= -1
        negativeFeeling = True

    elif ball.xcor() < -350:
        score_b += 1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a, score_b), align="center", font=("Courier", 24, "normal"))
        ball.goto(0, 0)
        ball.dx *= -1
        negativeFeeling = True

    # Paddle and ball collisions
    if ball.xcor() < -340 and ball.ycor() < paddle_a.ycor() + 50 and ball.ycor() > paddle_a.ycor() - 50:
        ball.dx *= -1
        positiveFeeling = True

    elif ball.xcor() > 340 and ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() > paddle_b.ycor() - 50:
        ball.dx *= -1
        positiveFeeling = True

    # Now we call the encoders to create bit representations for each value, and encode them.
    paddleABits  = paddleAEncoder.encode( paddle_a.ycor() )
    paddleBBits  = paddleBEncoder.encode( paddle_b.ycor() )
    ballBitsX    = ballEncoderX.encode( ball.xcor() )
    ballBitsY    = ballEncoderY.encode( ball.ycor() )
    ballBitsVelX = ballEncoderVelX.encode( xSpeed )
    ballBitsVelY = ballEncoderVelY.encode( ySpeed )

    # Concatenate all these encodings into one large encoding for Spatial Pooling.
    encoding = SDR( encodingWidth ).concatenate([paddleABits, paddleBBits, ballBitsX, ballBitsY, ballBitsVelX, ballBitsVelY])

    # Create an SDR to represent active columns, This will be populated by the
    # compute method below. It must have the same dimensions as the Spatial Pooler.
    activeColumns = SDR( sp.getColumnDimensions() )

    # Execute Spatial Pooling algorithm over input space (with learning turned on).
    sp.compute(encoding, True, activeColumns)

    # Execute Temporal Memory algorithm over active mini-columns and get the active cells.
    tm.compute(activeColumns, learn=True)

    activeCells = tm.getActiveCells()

    # Adjust past SDR stored array, the most recent time step goes at the beginning.
    pastActive.insert(0, activeCells)
    while len(pastActive) > pastFeelingSize:
        del pastActive[-1]

    # Compute the present feeling state and radiate it backwards.
    currFeelingAdjust = feelingAdjust
    if negativeFeeling:
        for x in pastActive:
            for y in x:
                cellFeeling[y] -= currFeelingAdjust
                currFeelingAdjust * 0.5
                negativeFeeling = False
    elif positiveFeeling:
        currFeelingAdjust = feelingAdjust
        for x in pastActive:
            for y in x:
                cellFeeling[y] += currFeelingAdjust
                currFeelingAdjust * 0.5
                positiveFeeling = False

    # Find the 40 (numActiveColumnsPerInhArea) highest feeling predicted cells
    tm.activateDendrites( learn=False )
    predictedCells = tm.getPredictiveCells()
    highestFeeling = [[0.0, 0]] * 40                         # (feeling, cell)
    print(predictedCells)
    for a in predictedCells:
        highestFeeling.sort(key=lambda pair: pair[0])        # Sort list according to feeling
        if cellFeeling[a] > highestFeeling[0][0]:
            highestFeeling[0][0] = cellFeeling[a]
            highestFeeling[0][1] = a

    # Compute XOR of current SDR winnerCells SDR and predicted highestFeeling SDR
    winnerCells = tm.getWinnerCells()
    XORCells = set(winnerCells).symmetric_difference(predictedCells)

    # Create arrays to represent active columns of motor, all initially zero. This
    # will be populated by the compute method below. It must have the same
    # dimensions as the Spatial Pooler.
    activeCellsForMotor = numpy.array([1 if i in activeCells else 0 for i in range(1280)])
    activeXORCellsForMotor = numpy.array([1 if i in XORCells else 0 for i in range(1280)])
    encodingForMotor = numpy.concatenate([activeXORCellsForMotor, activeCellsForMotor])
    activeColumnsMotor = numpy.zeros(2562)

    # Execute Spatial Pooling algorithm over input space (with learning turned on).
    mn.compute(encodingForMotor, True, activeColumnsMotor)
    activeColumnIndicesMotor = numpy.nonzero(activeColumnsMotor)[0]

    # Sum active columns mod 3 to find highest active, move paddles in this direction.
    upMotor = 0
    downMotor = 0
    stillMotor = 0
    for i in activeColumnIndicesMotor:
        if i % 3 == 0:
            upMotor += 1
        elif i % 3 == 1:
            downMotor += 1
        elif i % 3 == 2:
            stillMotor += 1

    # Move both paddles in accordance with Motor network, and adjust permanances to learn
    testList = []
    if learning == 0:
        if upMotor > downMotor and upMotor > stillMotor:
            paddle_a_up()
            paddle_b_up()
            for i in activeColumnIndicesMotor:
                if i % 3 == 0:
                    mn.getPermanence(i, testList)
                    mn.setPermanence(i, numpy.array([x + mn.getSynPermActiveInc() for x in testList]))
                elif i % 3 == 1:
                    mn.getPermanence(i, testList)
                    mn.setPermanence(i, numpy.array([x - mn.getSynPermInactiveDec() for x in testList]))
                elif i % 3 == 2:
                    mn.getPermanence(i, testList)
                    mn.setPermanence(i, numpy.array([x - mn.getSynPermInactiveDec() for x in testList]))
        elif downMotor > upMotor and downMotor > stillMotor:
            paddle_a_down()
            paddle_b_down()
            for i in activeColumnIndicesMotor:
                if i % 3 == 0:
                    mn.getPermanence(i, testList)
                    mn.setPermanence(i, numpy.array([x - mn.getSynPermInactiveDec() for x in testList]))
                elif i % 3 == 1:
                    mn.getPermanence(i, testList)
                    mn.setPermanence(i, numpy.array([x + mn.getSynPermActiveInc() for x in testList]))
                elif i % 3 == 2:
                    mn.getPermanence(i, testList)
                    mn.setPermanence(i, numpy.array([x - mn.getSynPermInactiveDec() for x in testList]))
        else:
            for i in activeColumnIndicesMotor:
                if i % 3 == 0:
                    mn.getPermanence(i, testList)
                    mn.setPermanence(i, numpy.array([x - mn.getSynPermInactiveDec() for x in testList]))
                elif i % 3 == 1:
                    mn.getPermanence(i, testList)
                    mn.setPermanence(i, numpy.array([x - mn.getSynPermInactiveDec() for x in testList]))
                elif i % 3 == 2:
                    mn.getPermanence(i, testList)
                    mn.setPermanence(i, numpy.array([x + mn.getSynPermActiveInc() for x in testList]))
    elif learning == 1:
        for i in activeColumnIndicesMotor:
            if i % 3 == 0:
                mn.getPermanence(i, testList)
                mn.setPermanence(i, numpy.array([x - mn.getSynPermInactiveDec() for x in testList]))
            elif i % 3 == 1:
                mn.getPermanence(i, testList)
                mn.setPermanence(i, numpy.array([x - mn.getSynPermInactiveDec() for x in testList]))
            elif i % 3 == 2:
                mn.getPermanence(i, testList)
                mn.setPermanence(i, numpy.array([x + mn.getSynPermActiveInc() for x in testList]))
    elif learning == 2:
        for i in activeColumnIndicesMotor:
            if i % 3 == 0:
                mn.getPermanence(i, testList)
                mn.setPermanence(i, numpy.array([x + mn.getSynPermActiveInc() for x in testList]))
            elif i % 3 == 1:
                mn.getPermanence(i, testList)
                mn.setPermanence(i, numpy.array([x - mn.getSynPermInactiveDec() for x in testList]))
            elif i % 3 == 2:
                mn.getPermanence(i, testList)
                mn.setPermanence(i, numpy.array([x - mn.getSynPermInactiveDec() for x in testList]))
    elif learning == 3:
        for i in activeColumnIndicesMotor:
            if i % 3 == 0:
                mn.getPermanence(i, testList)
                mn.setPermanence(i, numpy.array([x - mn.getSynPermInactiveDec() for x in testList]))
            elif i % 3 == 1:
                mn.getPermanence(i, testList)
                mn.setPermanence(i, numpy.array([x + mn.getSynPermActiveInc() for x in testList]))
            elif i % 3 == 2:
                mn.getPermanence(i, testList)
                mn.setPermanence(i, numpy.array([x - mn.getSynPermInactiveDec() for x in testList]))
