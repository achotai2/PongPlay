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

class MyPower:
    def __init__( self ):
        self.meActionA = 0
        self.meActionB = 0

    def learn_or_no_b( self ):
        if self.meActionB == 0:
            self.meActionB = 2
        elif self.meActionB == 2:
            self.meActionB = 0

    def moveBUp( self ):
        if self.meActionB != 0:
            self.meActionB = 1

myPower = MyPower()

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

    MCThreshold    = 0.5               # Threshold for movement classifier
    OCThreshold    = 0.5               # Threshold for origin point classifier if origin point is close
    OCMaxThreshold = 0.95              # Threshold when checking if this is an origin point

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

        # Set up Spatial Pooler and Temporal Memory
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

        self.tm = TemporalMemory(
            columnDimensions          = (3699,),
            cellsPerColumn            = 12,
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

        self.motorClassifier = Classifier( alpha=0.1 )               # Detects what motor functions available in this state.
        self.originPointClassifier = Classifier( alpha=0.1 )         # Detects if an origin point is nearby.
        self.motorCategories  = { "None": 0, "Up": 1, "Still": 2, "Down": 3 }     # Dictionary of movement vectors.
        self.originPointStore = []              # Array for storing detected origin points. Index is ID, value is Feeling

        self.feeling = 0

    def EncodeData( self, yPos, ballX, ballY, ballXSpeed, ballYSpeed, motorInput ):
        # Now we call the encoders to create bit representations for each value, and encode them.
        paddleBits  = self.paddleEncoder.encode( yPos )
        ballBitsX    = self.ballEncoderX.encode( ballX )
        ballBitsY    = self.ballEncoderY.encode( ballY )
        ballBitsVelX = self.ballEncoderVelX.encode( ballXSpeed )
        ballBitsVelY = self.ballEncoderVelY.encode( ballYSpeed )
        if motorInput != 0:                                             # motorInput = 0 means no input
            motorBits = self.motorEncoder.encode( motorInput )
        else:
            motorBits = SDR ( self.motorEncoder.size )                  # Encode empty SDR

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( self.encodingWidth ).concatenate([paddleBits, ballBitsX, ballBitsY, ballBitsVelX, ballBitsVelY, motorBits])
        return encoding

    def TemporalPredictor( self, encoding, learning ):
        # Create an SDR to represent active columns, This will be populated by the
        # compute method below. It must have the same dimensions as the Spatial Pooler.
        activeColumns = SDR( self.sp.getColumnDimensions() )

        # Execute Spatial Pooling algorithm over input space.
        self.sp.compute(encoding, learning, activeColumns)

        # Execute Temporal Memory algorithm over active mini-columns and get the active cells.
        self.tm.compute(activeColumns, learn=learning)

        activeCells = self.tm.getActiveCells()

        return activeCells

    def Hippocampus( self, ID, inputAction, yPos, ballX, ballY, ballXSpeed, ballYSpeed ):
        # Perform learning on TP previous time step motor output by:
        # Reset TP.
        self.tm.reset()
        # Run TP on last time step sense input and motor output (stored in buffer), with learning.
        if len( self.buffer ) > 0:
            encoding = self.EncodeData( self.buffer[0][0], self.buffer[0][1],
                self.buffer[0][2], self.buffer[0][3], self.buffer[0][4],
                self.buffer[0][5] )
            activeCells = self.TemporalPredictor( encoding, True )
            # Run motorClassifier, with learning, feeding it stored motor output.
            self.motorClassifier.learn ( activeCells, self.buffer[0][5] )
        # Run TP with current sense data without motor input, with learning:
        encoding = self.EncodeData( yPos, ballX, ballY, ballXSpeed, ballYSpeed, 0 )
        activeCells = self.TemporalPredictor( encoding, True )
        # Run MC with learning.
        self.motorClassifier.learn ( activeCells, 0 )

        # Reset temporal memory to signify start of new sequence.
        # All columns will burst, allowing us to get ranged predictions.
        self.tm.reset()

        # Produce motor output:
        motorScore = [ 0.0, 0.0, 0.0, 0.0 ]         # Keeps track of each motor output weighted score
        if inputAction == 0:
            # Using TemporalPredictor result,
            # check motorClassifier in inference mode to find possible motor outputs above some threshold.
            for idxMC, valMC in enumerate( self.motorClassifier.infer( activeCells ) ):
                if valMC >= self.MCThreshold:
                    # Run TemporalPredictor again with present sense date,
                    # without learning, with that motor input.
                    encoding = self.EncodeData( yPos, ballX, ballY, ballXSpeed, ballYSpeed, idxMC )
                    activeCells = self.TemporalPredictor( encoding, False )

                    # Check originPointClassifier in inference mode for detected origin points above some threshold.
                    for idxOC, valOC in enumerate( self.originPointClassifier.infer( activeCells ) ):
                        if valOC >= self.OCThreshold:
                            # Give the motor output a score, a weighted sum of (positive or negative depending
                            # on detected origin point value) and weighted by the strength observed by OC for
                            # each origin point. So, if one motor output sees: +A 70%, -B 60%, C 50%, it would
                            # get a score of +7-6+5=6.
                            motorScore[idxMC] += self.originPointStore[idxOC] * valOC
        elif inputAction == 1:
            motorScore[1] = 1.0
        elif inputAction == 2:
            motorScore[2] = 1.0
        elif inputAction == 3:
            motorScore[3] = 1.0

        # Check physical state of agent for strong input of positive or negative feeling.
        # If one exists:
        if self.feeling > 0 or self.feeling < 0:
            # Run TemporalPredictor again without motor input or learning.
            encoding = self.EncodeData( yPos, ballX, ballY, ballXSpeed, ballYSpeed, 0 )
            activeCells = self.TemporalPredictor( encoding, False )

            # Check if this is a stored origin point by checking activeCells from TemporalPredictor
            # with originPointClassifier, and checking if stored feeling is the same and current.
            isStored = False
            OPValue = 0
            for idxOC2, valOC2 in enumerate( self.originPointClassifier.infer( activeCells ) ):
                if valOC2 >= self.OCMaxThreshold and self.originPointStore[idxOC2] == self.feeling:
                    isStored = True
                    OPValue = idxOC2
            if isStored == False:
                # If it’s not stored then create a new origin point with a unique identification,
                # and current feeling state attached. Learn this origin point in originPointClassifier.
                self.originPointStore.append(self.feeling)
                OPValue = len(self.originPointStore) - 1
                for i in range(0, 99):
                    self.originPointClassifier.learn ( activeCells, len(self.originPointStore) - 1 )

            # Reset TP to indicate start of a new sequence.
            self.tm.reset()

            # Performing learning on this origin point in the buffer,
            # starting from tempPoint at furthest in the past:
            tempPoint = self.bufferSize - 1

            while tempPoint >= 0:
                # Run TemporalPredictor, with learning on, with buffer stored sense and motor input.
                # Feed input from tempPoint in buffer to present.
                encoding = self.EncodeData( self.buffer[tempPoint][0], self.buffer[tempPoint][1],
                    self.buffer[tempPoint][2], self.buffer[tempPoint][3], self.buffer[tempPoint][4],
                    self.buffer[tempPoint][5] )
                activeCells = self.TemporalPredictor( encoding, True )

                # Run motorClassifier, with learning, feeding it stored motor output.
                self.motorClassifier.learn ( activeCells, self.buffer[tempPoint][5] )

                # Run OC with learning feeding it this origin point.
                for x in range( self.bufferSize - tempPoint ):
                    self.originPointClassifier.learn ( activeCells, OPValue )

                # Cycle through this multiple times so that learning is enforced,
                # but run on less time to present so that more recent events are enforced more.
                # Shift tempPoint forward towards present. Keep doing this until reach the present.
                tempPoint -= 1

        # Store sensory input for this time-step, and winning motor output,
        # in ongoing buffer, and delete old entry.
        bufferInsert = [ yPos, ballX, ballY, ballXSpeed, ballYSpeed, motorScore.index( max( motorScore ) ) ]
        self.buffer.insert(0, bufferInsert)
        while len(self.buffer) > self.bufferSize:
            del self.buffer[-1]

#        print (ID, self.originPointStore)

        # Return winning motor function of agent. If there's a tie then choose a random one.
        largest = []
        for i in range(len( motorScore )):
            if sorted(motorScore, reverse=True)[0] == motorScore[i]:
                largest.append(i)

        return random.choice(largest)

# Create play agents
leftAgent = Agent()
rightAgent = Agent()

# Main game loop
while True:
    wn.update()         # Screen update

    # Keyboard bindings
#    wn.listen()
#    wn.onkey( paddle_a_up(), "w" )
#    wn.onkey( paddle_a_down(), "s" )
#    wn.onkey( myPower.moveBUp(), "Up" )
#    wn.onkey( paddle_b_down(), "Down" )
#    wn.onkey( myPower.learn_or_no_b(), "b" )

#    print (myPower.meActionB)

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
        rightAgent.feeling = -1

    elif ball.xcor() < -350:
        # Ball falls off left side of screen. B gets a point.
        score_b += 1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a, score_b), align="center", font=("Courier", 24, "normal"))
        ball.goto(0, 0)
        ball.dx *= -1
        ball.dy *= random.choice([-1, 1])
        leftAgent.feeling = -1

    # Paddle and ball collisions
    if ball.xcor() < -340 and ball.ycor() < paddle_a.ycor() + 50 and ball.ycor() > paddle_a.ycor() - 50:
        # Ball hits paddle A
        ball.dx *= -1
        ball.goto( -340, ball.ycor() )
        leftAgent.feeling = 1

    elif ball.xcor() > 340 and ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() > paddle_b.ycor() - 50:
        # Ball hits paddle B
        ball.dx *= -1
        ball.goto( 340, ball.ycor() )
        rightAgent.feeling = 1

    # Run each agents learning algorithm and produce movement.
    leftMove = leftAgent.Hippocampus( 1, myPower.meActionA, paddle_a.ycor(), ball.xcor(), paddle_a.ycor() - ball.ycor(), ball.dx, ball.dy )
    rightMove = rightAgent.Hippocampus( 2, myPower.meActionB, paddle_b.ycor(), ball.xcor(), paddle_b.ycor() - ball.ycor(), ball.dx, ball.dy )

    if leftMove == 1:
        paddle_a_up()
    elif leftMove == 3:
        paddle_a_down()

    if rightMove == 1:
        paddle_b_up()
    elif rightMove == 3:
        paddle_b_down()

    if myPower.meActionA != 0:
        myPower.meActionA = 2
    if myPower.meActionB != 0:
        myPower.meActionB = 2
