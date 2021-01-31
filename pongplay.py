# Simple Pong game combined with Nupic AI, in Python 3
# By Anand Chotai

import csv
import datetime
import numpy
import yaml
import random
import time
from pynput import keyboard

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
pen.write(
    "Left Agent: {}% on {} events".format(0, 0 ),
    align = "center", font = ("Courier", 24, "normal")
)
pen.goto(0, 230)
pen.write(
    "Right Agent: {}% on {} events".format(0, 0 ),
    align = "center", font = ("Courier", 24, "normal")
)

class Agent:
    #----------------------------------------------------------------------
    testThreshold   = 30
    motorDimensions = 3
    synapseInc = 0.05
    synapseDec = 0.01

    # Set up encoder parameters
    paddleEncodeParams    = ScalarEncoderParameters()
    ballXEncodeParams     = ScalarEncoderParameters()
    ballYEncodeParams     = ScalarEncoderParameters()
    ballVelEncodeParams   = RDSE_Parameters()
    manInEncodeParams     = ScalarEncoderParameters()

    paddleEncodeParams.activeBits = 99
    paddleEncodeParams.radius     = 200
    paddleEncodeParams.clipInput  = False
    paddleEncodeParams.minimum    = -screenHeight / 2
    paddleEncodeParams.maximum    = screenHeight / 2
    paddleEncodeParams.periodic   = False

    ballXEncodeParams.activeBits = 21
    ballXEncodeParams.radius     = 40
    ballXEncodeParams.clipInput  = False
    ballXEncodeParams.minimum    = -screenWidth / 2
    ballXEncodeParams.maximum    = screenWidth / 2
    ballXEncodeParams.periodic   = False

    ballYEncodeParams.activeBits = 21
    ballYEncodeParams.radius     = 40
    ballYEncodeParams.clipInput  = False
    ballYEncodeParams.minimum    = -screenHeight
    ballYEncodeParams.maximum    = screenHeight
    ballYEncodeParams.periodic   = False

    ballVelEncodeParams.size       = 400
    ballVelEncodeParams.activeBits = 25
    ballVelEncodeParams.resolution = 0.1

    manInEncodeParams.activeBits = 10
    manInEncodeParams.radius     = 1
    manInEncodeParams.clipInput  = False
    manInEncodeParams.minimum    = -1
    manInEncodeParams.maximum    = 2
    manInEncodeParams.periodic   = False

    def __init__( self, name ):
        self.ID = name

        self.senseBuffer = []       # Memory of all [ senseSDR, winningMotor ] experienced this sequence.

        # Set up encoders
        self.paddleEncoder   = ScalarEncoder( self.paddleEncodeParams )
        self.ballEncoderX    = ScalarEncoder( self.ballXEncodeParams )
        self.ballEncoderY    = ScalarEncoder( self.ballYEncodeParams )
        self.ballEncoderVelX = RDSE( self.ballVelEncodeParams )
        self.ballEncoderVelY = RDSE( self.ballVelEncodeParams )
        self.manInEncoder    = ScalarEncoder( self.manInEncodeParams )

        self.encodingWidth = ( self.paddleEncoder.size + self.ballEncoderX.size + self.ballEncoderY.size +
            self.ballEncoderVelX.size + self.ballEncoderVelY.size + self.manInEncoder.size )

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
            seed                      = 42
        )

        self.motorSynapse = numpy.random.uniform(
            low = 0.0, high = 1.0,
            size = ( self.tp.getColumnDimensions()[ 0 ] * self.tp.getCellsPerColumn() * self.motorDimensions, )
        )

        self.manualInput = -1

        self.numEvents = 0
        self.numSuccess = 0

    def Clear ( self ):
    # Clear sense buffer.
        self.senseBuffer.clear()
        self.tp.reset()


    def SendSuggest ( self, input ):
    # Sets the movement suggestion from user. manualInput = -1 means no suggestion.
        self.manualInput = input
        self.tp.reset()

    def EncodeSenseData ( self, yPos, ballX, ballY, ballXSpeed, ballYSpeed ):
    # Encodes sense data as an SDR and returns it.

        # Now we call the encoders to create bit representations for each value, and encode them.
        paddleBits   = self.paddleEncoder.encode( yPos )
        ballBitsX    = self.ballEncoderX.encode( ballX )
        ballBitsY    = self.ballEncoderY.encode( ballY )
        ballBitsVelX = self.ballEncoderVelX.encode( ballXSpeed )
        ballBitsVelY = self.ballEncoderVelY.encode( ballYSpeed )
        manInBits    = self.manInEncoder.encode( self.manualInput )

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( self.encodingWidth ).concatenate( [ paddleBits, ballBitsX, ballBitsY, ballBitsVelX, ballBitsVelY, manInBits ] )
        senseSDR = SDR( self.sp.getColumnDimensions() )
        self.sp.compute( encoding, True, senseSDR )

        return senseSDR

    # NOT USED CURRENTLY
    def Overlap ( self, SDR1, SDR2 ):
    # Computes overlap score between two passed SDRs.

        overlap = 0

        for cell1 in SDR1.sparse:
            if cell1 in SDR2.sparse:
                overlap += 1

        return overlap

    # NOT USED CURRENTLY
    def GreatestOverlap ( self, testSDR, listSDR, threshold ):
    # Finds SDR in listSDR with greatest overlap with testSDR and returns it, and its index in the list.
    # If none are found above threshold or if list is empty it returns an empty SDR of length testSDR, with index -1.

        greatest = [ SDR( testSDR.size ), -1 ]

        if len(listSDR) > 0:
            # The first element of listSDR should always be a union of all the other SDRs in list,
            # so a check can be performed first.
            if self.Overlap( testSDR, listSDR[0] ) >= threshold:
                aboveThreshold = []
                for idx, checkSDR in enumerate( listSDR ):
                    if idx != 0:
                        thisOverlap = self.Overlap( testSDR, checkSDR )
                        if thisOverlap >= threshold:
                            aboveThreshold.append( [ thisOverlap, [checkSDR, idx] ] )
                if len( aboveThreshold ) > 0:
                    greatest = sorted( aboveThreshold, key = lambda tup: tup[0], reverse = True )[ 0 ][ 1 ]

        return greatest

    def DetermineBurstPercent ( self ):
    # Calculates percentage of active cells that are currently bursting.

        activeCellsTP = self.tp.getActiveCells()

        # Get columns of all active cells.
        activeColumnsTP = []
        for cCell in activeCellsTP.sparse:
            activeColumnsTP.append( self.tp.columnForCell( cCell ) )

        # Get count of active cells in each active column.
        colUnique, colCount = numpy.unique( activeColumnsTP, return_counts = True )

        # Compute percentage of columns that are bursting.
        bursting = 0
        for c in colCount:
            if c > 1:
                bursting += 1
        burstPercent = int( 100 * bursting / len( colCount ) )

        return burstPercent

    def Hippocampus ( self, feeling, sequenceLength ):
    # Learns sequence back sequenceLength-time steps in memory, then stores sequence along with feeling.

        if feeling > 1.0 or feeling < -1.0:
            sys.exit( "Feeling states should be in the range [-1.0, 1.0]" )

        if len( self.senseBuffer ) == 0:
            return None
        elif len( self.senseBuffer ) < sequenceLength:
            learnRange = len( self.senseBuffer )
        else:
            learnRange = sequenceLength

        self.tp.reset()
        for idx in range( learnRange ):
            # Learn sequence in tp back sequenceLength-time steps.
            self.tp.compute( self.senseBuffer[ learnRange - idx - 1 ][ 0 ], learn = True )
            self.tp.activateDendrites( learn = True )
            winnerCellsTP = self.tp.getWinnerCells()

            # Train motor connections based on winner cells and remembered winningMotor.
            for cell in winnerCellsTP.sparse:
                for i in range( self.motorDimensions ):
                    if i == self.senseBuffer[ learnRange - idx - 1 ][ 1 ]:
                        self.motorSynapse[ ( cell * self.motorDimensions ) + i ] += self.synapseInc * feeling
                        if self.motorSynapse[ ( cell * self.motorDimensions ) + i ] > 1.0:
                            self.motorSynapse[ ( cell * self.motorDimensions ) + i ] = 1.0
                    else:
                        self.motorSynapse[ ( cell * self.motorDimensions ) + i ] -= self.synapseDec * feeling
                        if self.motorSynapse[ ( cell * self.motorDimensions ) + i ] < 0.0:
                            self.motorSynapse[ ( cell * self.motorDimensions ) + i ] = 0.0

    def Brain ( self, yPos, ballX, ballY, ballXSpeed, ballYSpeed ):
    # Agents brain center.

        # Generate SDR for sense data by feeding sense data into SP with learning.
        senseSDR = self.EncodeSenseData( yPos, ballX, ballY, ballXSpeed, ballYSpeed )

        # Feed present senseSDR into tp and generate active cells.
        self.tp.compute( senseSDR, learn = True )
        self.tp.activateDendrites( learn = True )
        winnerCellsTP = self.tp.getWinnerCells()

        if self.DetermineBurstPercent() <= 5:
            print("Less than 5% burst")

        motorScore = [ 0.0, 0.0, 0.0 ]         # Keeps track of each motor output weighted score [ UP, STILL, DOWN ]
        # Use active cells to determine motor action by feeding through motorSynapse.
        for cell in winnerCellsTP.sparse:
            for i in range( self.motorDimensions ):
                motorScore[i] += self.motorSynapse[ ( cell * self.motorDimensions ) + i ]

        # Use largest motorScore to choose winningMotor action.
        largest = []
        for i, v in enumerate( motorScore ):
            if sorted( motorScore, reverse=True )[ 0 ] == v:
                largest.append( i )                                         # 0 = UP, 1 = STILL, 2 = DOWN
        winningMotor = random.choice( largest )

        # Add senseSDR and winningMotor to buffer.
        self.senseBuffer.insert( 0, [ senseSDR, winningMotor ] )

        if self.manualInput != -1:
            # If motor suggestion, manualInput, equals winningMotor then send a small reward.
            if winningMotor == self.manualInput:
                self.Hippocampus( 0.1, 2 )
            # If not send a small punishment.
            else:
                self.Hippocampus( -0.1, 2 )

        # Return winning motor function.
        return winningMotor

        #----------------------------------------------------------------------

# Create play agents
leftAgent = Agent( 'Left' )
rightAgent = Agent( 'Right' )

# Functions
def ReDrawScore():
    if leftAgent.numEvents == 0:
        leftPercent = 0.0
    else:
        leftPercent  = int( 100 * leftAgent.numSuccess / leftAgent.numEvents )
    if rightAgent.numEvents == 0:
        rightPercent = 0.0
    else:
        rightPercent = int( 100 * rightAgent.numSuccess / rightAgent.numEvents )

    pen.clear()
    pen.goto(0, 260)
    pen.write(
        "Left Agent: {}% on {} events".format(
            leftPercent,
            leftAgent.numEvents,
        ),
        align = "center", font = ("Courier", 24, "normal")
    )
    pen.goto(0, 230)
    pen.write(
        "Right Agent: {}% on {} events".format(
            rightPercent,
            rightAgent.numEvents,
        ),
        align = "center", font = ("Courier", 24, "normal")
    )

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

def on_press( key ):
    if str(key) == ( 'Key.esc' ):
        quit()
    elif str( key ) == ( 'w' ):
        leftAgent.SendSuggest( 0 )
    elif str( key ) == ( 's' ):
        leftAgent.SendSuggest( 2 )

def on_release( key ):
    leftAgent.SendSuggest( -1 )

listener = keyboard.Listener( on_press = on_press, on_release = on_release )
listener.start()

while True:
# Main game loop

    wn.update()         # Screen update

    # Move the ball
    ball.setx( ball.xcor() + ball.dx )
    ball.sety( ball.ycor() + ball.dy )

    # Border checking top and bottom.
    if ball.ycor() > 290:
        ball.sety( 290 )
        ball.dy *= -1

    elif ball.ycor() < -290:
        ball.sety( -290 )
        ball.dy *= -1

    # Border checking left and right.
    if ball.xcor() > 350:
        # Ball falls off right side of screen. A gets a point.
        p = -( numpy.arctan( ( numpy.absolute( paddle_b.ycor() - ball.ycor() ) - ( screenHeight / 2 ) ) / 10 ) / numpy.pi ) - 0.5
        rightAgent.Hippocampus( p, 5 )
        leftAgent.Clear()
        rightAgent.Clear()

        rightAgent.numEvents += 1
        ReDrawScore()

        ball.goto(0, 0)
        ball.dx *= -1
        ball.dy *= random.choice( [ -1, 1 ] )

    elif ball.xcor() < -350:
        # Ball falls off left side of screen. B gets a point.
        p = -( numpy.arctan( ( numpy.absolute( paddle_a.ycor() - ball.ycor() ) - ( screenHeight / 2 ) ) / 10 ) / numpy.pi ) - 0.5
        leftAgent.Hippocampus( p, 5 )
        leftAgent.Clear()
        rightAgent.Clear()

        leftAgent.numEvents += 1
        ReDrawScore()

        ball.goto(0, 0)
        ball.dx *= -1
        ball.dy *= random.choice( [ -1, 1 ] )

    # Paddle and ball collisions
    if ball.xcor() < -340 and ball.ycor() < paddle_a.ycor() + 50 and ball.ycor() > paddle_a.ycor() - 50:
        # Ball hits paddle A
        leftAgent.Hippocampus( 1.0, 20 )
        leftAgent.Clear()
        rightAgent.Clear()

        leftAgent.numSuccess += 1
        leftAgent.numEvents += 1
        ReDrawScore()

        ball.dx *= -1
        ball.goto( -340, ball.ycor() )

    elif ball.xcor() > 340 and ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() > paddle_b.ycor() - 50:
        # Ball hits paddle B
        rightAgent.Hippocampus( 1.0, 20 )
        leftAgent.Clear()
        rightAgent.Clear()

        rightAgent.numSuccess += 1
        rightAgent.numEvents += 1
        ReDrawScore()

        ball.dx *= -1
        ball.goto( 340, ball.ycor() )

    # Run each agents learning algorithm and produce movement.
    leftMove = leftAgent.Brain( paddle_a.ycor(), ball.xcor(), paddle_a.ycor() - ball.ycor(), ball.dx, ball.dy )
    rightMove = rightAgent.Brain( paddle_b.ycor(), ball.xcor(), paddle_b.ycor() - ball.ycor(), ball.dx, ball.dy )

    # Move agents.
    if leftMove == 0:
        paddle_a_up()
    elif leftMove == 2:
        paddle_a_down()

    if rightMove == 0:
        paddle_b_up()
    elif rightMove == 2:
        paddle_b_down()
