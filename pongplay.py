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
    bufferSize = 20                     # Size of buffer of past input states to store.

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
        self.motorEncoder    = ScalarEncoder ( self.motorEncodeParams )

        self.encodingWidth = ( self.paddleEncoder.size + self.ballEncoderX.size + self.ballEncoderY.size +
            self.ballEncoderVelX.size + self.ballEncoderVelY.size + self.motorEncoder.size )

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
            inputDimensions            = (3699,),
            columnDimensions           = (100,),
            potentialPct               = 0.85,
            potentialRadius            = 3699,
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

        self.tm = TemporalMemory(
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

        self.motorClassifier = Classifier( alpha=0.1 )               # Detects what motor functions available in this state.

    def EncodeData( self, yPos, ballX, ballY, ballXSpeed, ballYSpeed, motorInput ):
        # Now we call the encoders to create bit representations for each value, and encode them.
        paddleBits  = self.paddleEncoder.encode( yPos )
        ballBitsX    = self.ballEncoderX.encode( ballX )
        ballBitsY    = self.ballEncoderY.encode( ballY )
        ballBitsVelX = self.ballEncoderVelX.encode( ballXSpeed )
        ballBitsVelY = self.ballEncoderVelY.encode( ballYSpeed )
        motorBits = self.motorEncoder.encode( motorInput )

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( self.encodingWidth ).concatenate( [ paddleBits, ballBitsX, ballBitsY, ballBitsVelX, ballBitsVelY, motorBits ] )

        return encoding

    def Hippocampus( self, ID, feeling, yPos, ballX, ballY, ballXSpeed, ballYSpeed ):
        # Agents brain center.

        # If a feeling state has been triggered:
        if feeling != 0:
            # Go through buffer and teach temporal memory network this jumping step.
            self.tm.reset()

            for buffEntry in self.buffer:
                encoding = self.EncodeData( buffEntry[ 0 ], buffEntry[ 1 ], buffEntry[ 2 ], buffEntry[ 3 ], buffEntry[ 4 ], buffEntry[ 5 ] )
            activeColumns = SDR( self.sp.getColumnDimensions() )
            self.sp.compute( encoding, True, activeColumns )
            self.tm.compute(activeColumns, True)
            activeCells = self.tm.getActiveCells()

            encoding = self.EncodeData( yPos, ballX, ballY, ballXSpeed, ballYSpeed, 2 )
            activeColumns = SDR( self.sp.getColumnDimensions() )
            self.sp.compute( encoding, True, activeColumns )
            self.tm.compute(activeColumns, True)
            activeCellsTM = self.tm.getActiveCells()

            # Also teach the classifier this feeling state.
            if feeling < 0:
                self.motorClassifier.learn( activeCellsTM, 0 )
            elif feeling > 0:
                self.motorClassifier.learn( activeCellsTM, 1 )

            # Clear the buffer as we're starting a new sequence.
            self.buffer.clear()

            return 2

        # No feeling state:
        else:
            # Run SP with current sense data without motor input, with learning:
            encoding = self.EncodeData( yPos, ballX, ballY, ballXSpeed, ballYSpeed, 2 )
            activeColumns = SDR( self.sp.getColumnDimensions() )
            self.sp.compute( encoding, True, activeColumns )

            # Choose motor output:
            # Feed SP activated cells to motor network.
            activeMotorColumns = SDR( self.mn.getColumnDimensions() )
            self.mn.compute( activeColumns, True, activeMotorColumns )

            # Choose winning motor action by adding up points for activated cells.
            motorScore = [ 0, 0, 0 ]         # Keeps track of each motor output weighted score UP, STILL, DOWN
            for idxMC in activeMotorColumns.sparse:
                motorScore[ idxMC % 3 ] += 1
            largest = []
            for i, v in enumerate( motorScore ):
                if sorted( motorScore, reverse=True )[ 0 ] == motorScore[ i ]:
                    largest.append( i + 1 )                                         # 1 = UP, 2 = STILL, 3 = DOWN
            winningMotor = random.choice( largest )

            # Plug in sense data and winningMotor to buffer.
            bufferInsert = [ yPos, ballX, ballY, ballXSpeed, ballYSpeed, winningMotor ]
            self.buffer.insert( 0, bufferInsert )
            while len( self.buffer ) > self.bufferSize:
                del self.buffer[ -1 ]

            # Plug in winningMotor to temporal memory network and get feeling of predicted end state.
            self.tm.reset()
            encoding = self.EncodeData( yPos, ballX, ballY, ballXSpeed, ballYSpeed, winningMotor )
            activeColumns = SDR( self.sp.getColumnDimensions() )
            self.sp.compute( encoding, learn=False, activeColumns )
            self.tm.compute( activeColumns, learn=False )
            self.tm.activateDendrites( learn=False )
            activeCellsTM = self.tm.getPredictiveCells()

            # Adjust connections between sp and mn.
            if len( self.motorClassifier.infer( activeCellsTM ) ) >= 2:
                for motCell in activeMotorColumns.sparse:       # Go through all the active motor cells in this time step...
                    permanence = numpy.zeros( 3699, dtype=numpy.float32 )
                    self.mn.getPermanence( motCell, permanence, 0.0 )   # and get their strength of connections to the sp cells.
                    for actCell in activeColumns.sparse:         # Get all the active sp cells in this time step...
                        # and adjust their connection strength to this active motor cell accordingly.
                        # (connection strength should vary between -1 and 1, so we use a tanh function)
                        permVal = numpy.tan( permanence[ actCell ] )
                        if self.motorClassifier.infer( activeCellsTM )[ 0 ] < self.motorClassifier.infer( activeCellsTM )[ 1 ]:
                            permVal += self.mn.getSynPermActiveInc()
                        else:
                            permVal -= self.mn.getSynPermInactiveDec()
                        permanence[ actCell ] = numpy.tanh( permVal )

                    self.mn.setPermanence( motCell, permanence )

            # Return winning motor function of agent.
            return winningMotor

# Create play agents
leftAgent = Agent()
rightAgent = Agent()

# Main game loop
while True:
    leftAgentFeeling = 0
    rightAgentFeeling = 0

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
        # Ball falls off right side of screen. A gets a point.
        score_a += 1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a, score_b), align="center", font=("Courier", 24, "normal"))
        leftAgentFeeling = 1
        rightAgentFeeling = -1
#        rightAgentFeeling = numpy.exp( -numpy.absolute( ball.ycor() - paddle_b.ycor() ) / 200 ) - 1
        ball.goto(0, 0)
        ball.dx *= -1
        ball.dy *= random.choice([-1, 1])

    elif ball.xcor() < -350:
        # Ball falls off left side of screen. B gets a point.
        score_b += 1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a, score_b), align="center", font=("Courier", 24, "normal"))
        leftAgentFeeling = -1
        rightAgentFeeling = 1
        ball.goto(0, 0)
        ball.dx *= -1
        ball.dy *= random.choice([-1, 1])

    # Paddle and ball collisions
    if ball.xcor() < -340 and ball.ycor() < paddle_a.ycor() + 50 and ball.ycor() > paddle_a.ycor() - 50:
        # Ball hits paddle A
        leftAgentFeeling = 1
        rightAgentFeeling = 0
        ball.dx *= -1
        ball.goto( -340, ball.ycor() )

    elif ball.xcor() > 340 and ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() > paddle_b.ycor() - 50:
        # Ball hits paddle B
        leftAgentFeeling = 0
        rightAgentFeeling = 1
        ball.dx *= -1
        ball.goto( 340, ball.ycor() )

    # Run each agents learning algorithm and produce movement.
    leftMove = leftAgent.Hippocampus( 1, leftAgentFeeling, paddle_a.ycor(), ball.xcor(), paddle_a.ycor() - ball.ycor(), ball.dx, ball.dy )
    rightMove = rightAgent.Hippocampus( 2, rightAgentFeeling, paddle_b.ycor(), ball.xcor(), paddle_b.ycor() - ball.ycor(), ball.dx, ball.dy )

    if leftMove == 1:
        paddle_a_up()
    elif leftMove == 3:
        paddle_a_down()

    if rightMove == 1:
        paddle_b_up()
    elif rightMove == 3:
        paddle_b_down()
