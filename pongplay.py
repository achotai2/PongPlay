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
    motorThreshold = 20                     # Threshold for overlap for recognized motor output in motorStore.
    goalThreshold = 20
    motorSDRsize = 30                       # Size of SDRs stored in motorStore
    goalSDRsize = 30
    maxTimeBetGoal = 25                     # The maximum time allowed between goal states, more than this we add one

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
        self.senseBuffer = []                         # Buffer of past input sense data state SDRs.
        self.motorBuffer = []                         # Stores a tuple of MP input and winning motor output SDR.
        self.motorStore = []                          # Stores winning SDRs used to produce motor output.
        self.goalStore = []                           # Stores SDRs of all recognized goal states, and integer for count, and time last seen.

        # Set up encoders
        self.paddleEncoder   = ScalarEncoder( self.paddleEncodeParams )
        self.ballEncoderX    = ScalarEncoder( self.ballXEncodeParams )
        self.ballEncoderY    = ScalarEncoder( self.ballYEncodeParams )
        self.ballEncoderVelX = RDSE( self.ballVelEncodeParams )
        self.ballEncoderVelY = RDSE( self.ballVelEncodeParams )
#        self.motorEncoder    = ScalarEncoder ( self.motorEncodeParams )

        self.encodingWidth = ( self.paddleEncoder.size + self.ballEncoderX.size + self.ballEncoderY.size +
            self.ballEncoderVelX.size + self.ballEncoderVelY.size )

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

        self.mp = TemporalMemory(
            columnDimensions          = (67584,),
            cellsPerColumn            = 1,
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

        self.tp = TemporalMemory(
            columnDimensions          = (2048,),
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

    def ClearBuffer ( self ):
    # Clears both buffers.

        self.motorBuffer.clear()
        self.senseBuffer.clear()

    def EncodeSenseData ( self, yPos, ballX, ballY, ballXSpeed, ballYSpeed ):
    # Encodes sense data as an SDR and returns it.

        # Now we call the encoders to create bit representations for each value, and encode them.
        paddleBits  = self.paddleEncoder.encode( yPos )
        ballBitsX    = self.ballEncoderX.encode( ballX )
        ballBitsY    = self.ballEncoderY.encode( ballY )
        ballBitsVelX = self.ballEncoderVelX.encode( ballXSpeed )
        ballBitsVelY = self.ballEncoderVelY.encode( ballYSpeed )
#        motorBits = self.motorEncoder.encode( motorInput )

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( self.encodingWidth ).concatenate( [ paddleBits, ballBitsX, ballBitsY, ballBitsVelX, ballBitsVelY ] )
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    def Overlap ( self, SDR1, SDR2 ):
    # Computes overlap score between two passed SDRs.

        overlap = 0

        for cell1 in SDR1.sparse:
            for cell2 in SDR2.sparse:
                if cell1 == cell2:
                    overlap += 1

        return overlap

    def GreatestOverlap ( self, testSDR, listSDR, threshold ):
    # Finds SDR in listSDR with greatest overlap with testSDR and returns it, and its index in the list.
    # If none are found above threshold or if list is empty it returns an empty SDR of length testSDR.

        aboveThreshold = []
        if listSDR:
            for idx, checkSDR in enumerate( listSDR ):
                thisOverlap = self.Overlap( testSDR, checkSDR )
                if thisOverlap >= threshold:
                    aboveThreshold.insert( 0, [ thisOverlap, [checkSDR, idx] ] )
        if aboveThreshold:
            greatest = sorted( aboveThreshold, key=lambda tup: tup[0], reverse=True )[ 0 ][ 1 ]
        else:
            greatest = [ SDR( testSDR.size ), -1 ]

        return greatest

    def PredictionNetRewardEvent ( self ):
    # Triggered when ball hits a paddle, trains the prediction network, PN

        # Go through sense data buffer (from oldest to newest) and compare each state to the goal SDR storage.
        # If any are found store, in order, any found goal SDR states. Also, if it's been a while since we
        # saw a goal state, add a new one to goal state storages. Also add occurrence integer to each found goal
        # state, and reset their time since last seen to 1.
        foundGoals = []
        timeBetGoal = 0
        for indx, senseSDR in enumerate( self.senseBuffer ):
            timeBetGoal += 1
            if self.goalStore:
                testSDR = self.GreatestOverlap( senseSDR, [ i[0] for i in self.goalStore ], self.goalThreshold )
                if testSDR[1] != -1:
                    foundGoals.insert( -1, testSDR )
                    self.goalStore[ testSDR[1] ][1] += 1
                    self.goalStore[ testSDR[1] ][2] += 1
                    timeBetGoal = 0
            if timeBetGoal > self.maxTimeBetGoal:
                x = random.randint( 0, 100 )
                y = numpy.power( ( x * numpy.cbrt( self.maxTimeBetGoal / 2 ) / 50 ) - numpy.cbrt( self.maxTimeBetGoal / 2 ), 3 ) + ( self.maxTimeBetGoal / 2 )
                insStep = int( numpy.rint( y ) )
                foundGoals.insert( -1, self.senseBuffer[ indx - insStep ] )
                self.goalStore.insert( -1, [ self.senseBuffer[ indx - insStep ], 1, 1 ] )
                timeBetGoal = insStep

        # Add last senseData state to foundGoals at the end. This is the ball hitting paddle event.
        foundGoals.insert( -1, self.senseBuffer[ -1 ] )

        print (foundGoals)

        # Train PM by:
        # Starting with the oldest entry of sense data buffer, choose one.
        nextFound = 0
        for senseDataSDR in self.senseBuffer:

            # Reset TP.
            self.tp.reset()

            # Feed in chosen entry to TP, with learning.
            self.tp.compute( senseDataSDR, True )
#            self.tp.activateDendrites( learn=True )                    DO I NEED TO RUN THIS?

            # Check the chosen entry SDR if it is next goal state found above.
            # If chosen entry is next found goal state feed subsequent found goal state into TP, with learning.
            # If chosen entry isn’t next found goal state then feed in next found goal state into TP, with learning.
            if self.Overlap( foundGoals[ nextFound ], senseDataSDR ) >= self.goalThreshold and nextFound + 1 != len( foundGoals ):
                nextFound += 1
            self.tp.compute( foundGoals[ nextFound ], True )

            # Repeat for all steps of buffer.

        # Goal state storage cleanup: Go through goal state storage and choose any goal states that haven’t
        # been seen in 50+ turns, of these delete the 25% with the lowest event count.
# DO THIS LATER

        # Clear sense data buffer.
        self.senseBuffer.clear()

    def HabitualNetRewardEvent ( self, winningGoalSDR ):
    # Triggered by PredictionNetwork when we successfully arrive at a goal state. Trains MP, habitual network.

        # Train MP by:
        # Starting with oldest entry of motor buffer, select one.
        for motorSDR in self.motorBuffer:

            # Reset MP.
            self.mp.reset()

            # Feed MP the chosen entry input, with learning.
            self.mp.compute( motorSDR[ 0 ], True )

            # Feed MP the chosen entry winning goal SDR, with learning.
            self.mp.compute( motorSDR[ 1 ], True )

            # Repeat above for all steps of buffer.

        # Clear motor buffer.
        self.motorBuffer.clear()

    def PredictionNetwork ( self, senseSDR ):
    # Predicts goal states in a temporally connected way.

        # Check if we’ve reached any goal SDR by comparing overlap of all stored goal SDRs with sense data SDR.
        # If any above threshold choose the one with highest overlap, run habitual network reward event.
        reachedGoal = self.GreatestOverlap( senseSDR, [ i[0] for i in self.goalStore ], self.goalThreshold )
        if reachedGoal[1] != -1:
            self.HabitualNetRewardEvent( reachedGoal )

        # Reset PN.
        self.tp.reset()

        # Plug sense data SDR into TP, without learning, to generate predicted cells.
        self.tp.compute( senseSDR, False )
        self.tp.activateDendrites( learn=False )
        predCellsTP = self.tp.getPredictiveCells()

        # Compare predicted cells against stored goal states. Choose best one above threshold as winning goal SDR.
        # If there are no stored goal states then make winning goal SDR empty.
        winningGoal = self.GreatestOverlap ( predCellsTP, [ i[0] for i in self.goalStore ], self.goalThreshold )

        # Return winning goal SDR.
        return winningGoal[0]

    def Hippocampus ( self, ID, feeling, yPos, ballX, ballY, ballXSpeed, ballYSpeed ):
    # Agents brain center.

        # Generate SDR for sense data by feeding sense data into SP with learning.
        senseSDR = self.EncodeSenseData( yPos, ballX, ballY, ballXSpeed, ballYSpeed )

        # Store sense data SDR into sense data buffer.
        self.senseBuffer.insert( -1, senseSDR )

        # Run prediction network, sending it sense data SDR.
        goalSDR = self.PredictionNetwork ( senseSDR )

        # Habitual network: ------------------

        # Reset MP.
        self.mp.reset()

        # Concatenate sense data SDR + goal SDR (returned by prediction network)
        # and feed result into MP, without learning, to produce predicted cells.
        concateSDR = SDR( 67584 ).concatenate( senseSDR, goalSDR )
        self.mp.compute( concateSDR, False )
        self.mp.activateDendrites( learn=False )
        predCellsMP = self.mp.getPredictiveCells()

        # Compare predicted cells for overlap against motor output storage against some threshold,
        # to produce winning SDR. If storage is empty, or not enough overlap threshold then
        # winning SDR = x-number of random predicted cells (where x is a predetermined
        # size of memory storage items).
        aboveThreshold = self.GreatestOverlap( predCellsMP, [ i[0] for i in self.motorStore ], self.motorThreshold )
        winningSDR = SDR( predCellsMP.size )
        if aboveThreshold[1] != -1:
# DOES THIS GET US THE BIGGEST? DOES IT WORK THE WAY IT SHOULD?
            winningSDR = aboveThreshold[0]
        else:
            if predCellsMP.sparse.size >= self.motorSDRsize:
                winningSDR.sparse = sorted( random.sample( predCellsMP.sparse, self.motorSDRsize ), reverse=False )
            else:
                winningSDR.sparse = random.sample( range( 0, predCellsMP.size ), self.motorSDRsize )

        # Add MP input SDR and winning SDR as single entry to motor buffer.
        self.motorBuffer.insert( -1, [ concateSDR, winningSDR ] )

        # Compute winning motor output from winning SDR through modular cell ID.
        motorScore = [ 0, 0, 0 ]         # Keeps track of each motor output weighted score UP, STILL, DOWN
        for idxMC in winningSDR.sparse:
            motorScore[ idxMC % 3 ] += 1
        largest = []
        for i, v in enumerate( motorScore ):
            if sorted( motorScore, reverse=True )[ 0 ] == motorScore[ i ]:
                largest.append( i + 1 )                                         # 1 = UP, 2 = STILL, 3 = DOWN
        winningMotor = random.choice( largest )

        # Return winning motor function.
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
    ball.setx( ball.xcor() + ball.dx )
    ball.sety( ball.ycor() + ball.dy )

    # Border checking
    # Top and bottom
    if ball.ycor() > 290:
        ball.sety( 290 )
        ball.dy *= -1

    elif ball.ycor() < -290:
        ball.sety( -290 )
        ball.dy *= -1

    # Left and right
    if ball.xcor() > 350:
        # Ball falls off right side of screen. A gets a point.
        score_a += 1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a, score_b), align="center", font=("Courier", 24, "normal"))
        ball.goto(0, 0)
        ball.dx *= -1
        ball.dy *= random.choice( [ -1, 1 ] )
        leftAgent.ClearBuffer()
        rightAgent.ClearBuffer()

    elif ball.xcor() < -350:
        # Ball falls off left side of screen. B gets a point.
        score_b += 1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a, score_b), align="center", font=("Courier", 24, "normal"))
        leftAgentFeeling = -1
        rightAgentFeeling = 1
        ball.goto(0, 0)
        ball.dx *= -1
        ball.dy *= random.choice( [ -1, 1 ] )
        leftAgent.ClearBuffer()
        rightAgent.ClearBuffer()

    # Paddle and ball collisions
    if ball.xcor() < -340 and ball.ycor() < paddle_a.ycor() + 50 and ball.ycor() > paddle_a.ycor() - 50:
        # Ball hits paddle A
        leftAgentFeeling = 1
        rightAgentFeeling = 0
        ball.dx *= -1
        ball.goto( -340, ball.ycor() )
        leftAgent.PredictionNetRewardEvent()

    elif ball.xcor() > 340 and ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() > paddle_b.ycor() - 50:
        # Ball hits paddle B
        leftAgentFeeling = 0
        rightAgentFeeling = 1
        ball.dx *= -1
        ball.goto( 340, ball.ycor() )
        rightAgent.PredictionNetRewardEvent()

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
