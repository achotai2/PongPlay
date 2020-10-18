# Simple Pong game combined with Nupic AI, in Python 3
# By Anand Chotai

import csv
import datetime
import numpy
import yaml
import random

from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
import htm.bindings.encoders
ScalarEncoder = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters
RDSE = htm.bindings.encoders.RDSE
RDSE_Parameters = htm.bindings.encoders.RDSE_Parameters
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.algorithms import Classifier

screenHeight = 600          # Used in setting up screen and encoders
screenWidth = 800

xSpeed = 1                   # Speed for ball
ySpeed = 1

score_a = 0                 # Used to keep track of score
score_b = 0

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

class Agent:
    bufferSize = 40                     # Size of buffer of past input states to store.

    # Set up encoder parameters
    paddleEncodeParams    = ScalarEncoderParameters()
    ballXEncodeParams     = ScalarEncoderParameters()
    ballYEncodeParams     = ScalarEncoderParameters()
    ballVelEncodeParams   = RDSE_Parameters()
    motorEncodeParams     = ScalarEncoderParameters()

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
    ballYEncodeParams.minimum    = -screenHeight
    ballYEncodeParams.maximum    = screenHeight
    ballYEncodeParams.periodic   = False

    ballVelEncodeParams.size       = 400
    ballVelEncodeParams.activeBits = 20
    ballVelEncodeParams.resolution = 0.1

    motorEncodeParams
    motorEncodeParams.activeBits = 41
    motorEncodeParams.radius     = 1
    motorEncodeParams.clipInput  = False
    motorEncodeParams.minimum    = 1
    motorEncodeParams.maximum    = 3
    motorEncodeParams.periodic   = False

    def __init__( self ):
        # Set up buffer list
        self.buffer = []                         # Buffer of past input states and motor output.

        # Set up encoders
        self.paddleEncoder   = ScalarEncoder( self.paddleEncodeParams )
        self.ballEncoderX    = ScalarEncoder( self.ballXEncodeParams )
        self.ballEncoderY    = ScalarEncoder( self.ballYEncodeParams )
        self.ballEncoderVelX = RDSE( self.ballVelEncodeParams )
        self.ballEncoderVelY = RDSE( self.ballVelEncodeParams )

        self.encodingWidth = ( self.paddleEncoder.size + self.ballEncoderX.size + self.ballEncoderY.size +
            self.ballEncoderVelX.size + self.ballEncoderVelY.size )

        # Set up Spatial Pooler
        self.sp = SpatialPooler(
            inputDimensions            = (self.encodingWidth,),
            columnDimensions           = (3699,),
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

        # Set up Motor Network
        self.mn = SpatialPooler(
            inputDimensions            = (self.encodingWidth,),
            columnDimensions           = (100,),
            potentialPct               = 0.85,
            potentialRadius            = self.encodingWidth,
            globalInhibition           = True,
            localAreaDensity           = 0,
            numActiveColumnsPerInhArea = 10,
            synPermInactiveDec         = 0.005,
            synPermActiveInc           = 0.04,
            synPermConnected           = 0.1,
            boostStrength              = 3.0,
            seed                       = -1,
            wrapAround                 = True
        )

    def EncodeData( self, yPos, ballX, ballY, ballXSpeed, ballYSpeed ):
        # Now we call the encoders to create bit representations for each value, and encode them.
        paddleBits  = self.paddleEncoder.encode( yPos )
        ballBitsX    = self.ballEncoderX.encode( ballX )
        ballBitsY    = self.ballEncoderY.encode( ballY )
        ballBitsVelX = self.ballEncoderVelX.encode( ballXSpeed )
        ballBitsVelY = self.ballEncoderVelY.encode( ballYSpeed )

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( self.encodingWidth ).concatenate( [ paddleBits, ballBitsX, ballBitsY, ballBitsVelX, ballBitsVelY ] )
        return encoding

    def GetSDR( self, encoding, ID, learning ):
        # Create an SDR to represent active columns, This will be populated by the
        # compute method below. It must have the same dimensions as the Spatial Pooler.
        if ID == "sp":
            activeColumns = SDR( self.sp.getColumnDimensions() )
            self.sp.compute(encoding, learning, activeColumns)
        elif ID == "mn":
            activeColumns = SDR( self.mn.getColumnDimensions() )
            self.mn.compute(encoding, learning, activeColumns)

        return activeColumns

    def FeelingTrig( self, feeling ):
        # Triggered when agent is fed a feeling state.

        if self.feeling > 0 or self.feeling < 0:
            # Go back through buffer and adjust connections between SP and motor
            # network accordingly, with decreasing strength further back in buffer.
            for ind, val in enumerate( self.buffer ):
                activeCells = val[0]
                activeMotorCells = val[1]
                for motCell in activeMotorCells.sparse:       # Go through all the active motor cells in this buffer time step...
                    permanance = []
                    mn.getPermanance( motCell, permanance )   # and get their strength of connections to the sp cells.
                    for spatCell in permanance:             # Go through all these connected cells...
                        for actCell in activeCells.sparse:
                            if spatCell == actCell:         # and check if any are active sp cells in this buffer time step.
                                # If they are then adjust their connection strength accordingly.
                                spatCell += ( feeling * 0.05 ) * ( self.bufferSize - ind )

                    mn.setPermanance( motCell, permanance )

    def Hippocampus( self, ID, yPos, ballX, ballY, ballXSpeed, ballYSpeed ):
        # Agents brain center.

        # Run SP with current sense data without motor input, with learning:
        encoding = self.EncodeData( yPos, ballX, ballY, ballXSpeed, ballYSpeed )
        activeCells = self.GetSDR( encoding, "sp", True )

        # Choose motor output:
        # Feed SP activated cells to motor network.
        encoding = self.EncodeData( yPos, ballX, ballY, ballXSpeed, ballYSpeed )
        activeMotorCells = self.GetSDR( encoding, "mn", True )

        # Choose winning motor action by adding up points for activated cells.
        motorScore = [ 0, 0, 0 ]         # Keeps track of each motor output weighted score UP, STILL, DOWN
        for idxMC in activeMotorCells.sparse:
            if idxMC % 3 == 0:
                motorScore[0] += 1
            elif idxMC % 3 == 1:
                motorScore[1] += 1
            elif idxMC % 3 == 2:
                motorScore[2] += 1
        largest = []
        for i, v in enumerate( motorScore ):
            if sorted(motorScore, reverse=True)[0] == motorScore[i]:
                largest.append(i)
        winningMotor = random.choice(largest)

        print (winningMotor)

        # Adjust connections between SP and motor network to enforce winning motor action slightly.
        # TRY THIS LATER AND SEE IF IT IMPROVES OR NOT

        # Update buffer with activated cells in SP and motor network, 0 being most recent.
        # Each buffer entry contains the cell activations of spatial pooler and motor network.
        bufferInsert = [ activeCells, activeMotorCells ]
        self.buffer.insert( 0, bufferInsert )
        while len(self.buffer) > self.bufferSize:
            del self.buffer[-1]

        # Return winning motor function of agent.
        return winningMotor

# Create play agents
leftAgent = Agent()
rightAgent = Agent()

# Main game loop
while True:
    wn.update()         # Screen update

    leftAgent.feeling  = 0        # Reset feeling states for agents
    rightAgent.feeling = 0

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
        # Ball falls off right side of screen. A gets a point.
        score_a += 1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a, score_b), align="center", font=("Courier", 24, "normal"))
        ball.goto(0, 0)
        ball.dx *= -1
        ball.dy *= random.choice([-1, 1])
        rightAgent.FeelingTrig( -1 )

    elif ball.xcor() < -350:
        # Ball falls off left side of screen. B gets a point.
        score_b += 1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a, score_b), align="center", font=("Courier", 24, "normal"))
        ball.goto(0, 0)
        ball.dx *= -1
        ball.dy *= random.choice([-1, 1])
        leftAgent.FeelingTrig( -1 )

    # Paddle and ball collisions
    if ball.xcor() < -340 and ball.ycor() < paddle_a.ycor() + 50 and ball.ycor() > paddle_a.ycor() - 50:
        # Ball hits paddle A
        ball.dx *= -1
        ball.goto( -340, ball.ycor() )
        leftAgent.FeelingTrig( 1 )

    elif ball.xcor() > 340 and ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() > paddle_b.ycor() - 50:
        # Ball hits paddle B
        ball.dx *= -1
        ball.goto( 340, ball.ycor() )
        rightAgent.FeelingTrig( 1 )

    # Run each agents learning algorithm and produce movement.
    leftMove = leftAgent.Hippocampus( 1, paddle_a.ycor(), ball.xcor(), paddle_a.ycor() - ball.ycor(), ball.dx, ball.dy )
    rightMove = rightAgent.Hippocampus( 2, paddle_b.ycor(), ball.xcor(), paddle_b.ycor() - ball.ycor(), ball.dx, ball.dy )

    if leftMove == 1:
        paddle_a_up()
    elif leftMove == 3:
        paddle_a_down()

    if rightMove == 1:
        paddle_b_up()
    elif rightMove == 3:
        paddle_b_down()
