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

class Agent:
    #----------------------------------------------------------------------
    testThreshold   = 30
    motorDimensions = 3

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
    ballVelEncodeParams.activeBits = 20
    ballVelEncodeParams.resolution = 0.1

    manInEncodeParams.activeBits = 10
    manInEncodeParams.radius     = 1
    manInEncodeParams.clipInput  = False
    manInEncodeParams.minimum    = -1
    manInEncodeParams.maximum    = 2
    manInEncodeParams.periodic   = False


    def __init__( self, name ):
        self.ID = name

        self.SDRList = []           # List of SDR's to check against present moment.
        self.seqList = []           # List of sequences I've observed from this point, indexes referenced as in SDRList.
                                    # Each element is a list of sequences, each element of form :
                                    #   [ x ][ 0 ] = Final feeling of this sequence, in range [ -1, 1 ].
                                    #   [ x ][ 1 ] = Next seen active cells in this sequence.

        self.senseBuffer = []       # Memory of all senseSDR's experienced this sequence.
        self.cellsBuffer = []       # Memory of all active cells this sequence, referenced as in senseBuffer.

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
            low = 0.0, high = 1.0, size = ( self.tp.getColumnDimensions()[ 0 ] * self.tp.getCellsPerColumn() * self.motorDimensions, )
        )

        self.manualInput = -1

    def Reset ( self ):
    # Resets temoral memory for start of new sequence, and clears senseSDR sequence buffer.
        self.tp.reset()
        self.senseBuffer.clear()
        self.cellsBuffer.clear()

    def SendSuggest ( self, input ):
    # Sets the movement suggestion from user. manualInput = -1 means no suggestion.
        self.manualInput = input

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

    def Overlap ( self, SDR1, SDR2 ):
    # Computes overlap score between two passed SDRs.

        overlap = 0

        for cell1 in SDR1.sparse:
            if cell1 in SDR2.sparse:
                overlap += 1

        return overlap

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

    def Hippocampus ( self, feeling ):
    # Stores sequence along with feeling.

        if feeling > 1.0 or feeling < -1.0:
            sys.exit( "Feeling states should be in the range [-1.0, 1.0]" )

        for idx, pastSDR in enumerate( self.senseBuffer ):
            # Check if we've already seen this SDR.
            greatestSDR = self.GreatestOverlap( pastSDR, self.SDRList, self.sp.getNumActiveColumnsPerInhArea() )

            if idx < len( self.senseBuffer ) - 1:
                # If we haven't then add the SDR and the next seen active cells in the sequence.
                if greatestSDR[ 1 ] == -1:
                    self.SDRList.append( pastSDR )

                    seqList = []
                    insertSeq = [ feeling, self.cellsBuffer[ idx + 1 ] ]
                    seqList.append( insertSeq )
                    self.seqList.append( seqList )

                # If we have seen the SDR then check if the next active cells in the sequence are stored.
                else:
                    for seq in self.seqList[ greatestSDR[ 1 ] ]:
                        if self.Overlap( seq[ 1 ], self.cellsBuffer[ idx + 1 ] ) >= self.testThreshold:
                            # If they are stored then do they have the same feeling?
                            if seq[ 0 ] != feeling:
                                # If not then store it as a new sequence.
                                insertSeq = [ feeling, self.cellsBuffer[ idx + 1 ] ]
                                self.seqList[ greatestSDR[ 1 ] ].append( insertSeq )
                        else:
                            # If they are not stored then add it as a new sequence.
                            insertSeq = [ feeling, self.cellsBuffer[ idx + 1 ] ]
                            self.seqList[ greatestSDR[ 1 ] ].append( insertSeq )

        print (len(self.SDRList))

    def Brain ( self, yPos, ballX, ballY, ballXSpeed, ballYSpeed ):
    # Agents brain center.

        # Generate SDR for sense data by feeding sense data into SP with learning.
        senseSDR = self.EncodeSenseData( yPos, ballX, ballY, ballXSpeed, ballYSpeed )

        # Feed senseSDR into tp to get predictive cells.
        self.tp.compute( senseSDR, learn = True )
        self.tp.activateDendrites( learn = True )
        actvCellsTP = self.tp.getActiveCells()
        predCellsTP = self.tp.getPredictiveCells()

        # Add senseSDR and predCellsTP to buffers.
        self.senseBuffer.append( senseSDR )
        self.cellsBuffer.append( actvCellsTP )

        # Check if we've seen any of these predictive cell sequences before.
        greatestSDR = self.GreatestOverlap( senseSDR, self.SDRList, self.testThreshold )

        # Given these known sequences which one led to the most positive outcome?
        greatestPositive = -1
        if greatestSDR[ 1 ] != -1:
            for idx, seq in enumerate( self.seqList[ greatestSDR[ 1 ] ] ):
                if greatestPositive != -1:
                    if self.seqList[ greatestSDR[ 1 ] ][ greatestPositive ][ 0 ] < seq[ 0 ]:
                        greatestPositive = idx
                else:
                    if seq[ 0 ] > 0:
                        greatestPositive = idx

        # If we've detected a positive sequence feed these predictive cells to motor.
        if greatestPositive != -1:
            predToMotor = self.seqList[ greatestSDR[ 1 ] ][ greatestPositive ][ 1 ]
        # If we didn't then feed all the predicted cells predCellsTP to motor.
        else:
            predToMotor = predCellsTP

        motorScore = [ 0.0, 0.0, 0.0 ]         # Keeps track of each motor output weighted score [ UP, STILL, DOWN ]
        # Use predToMotor to determine motor action by feeding through motorSynapse.
        for cell in predToMotor.sparse:
            for i in range( self.motorDimensions ):
                motorScore[i] += self.motorSynapse[ ( cell * self.motorDimensions ) + i ]

        # Use largest motorScore to choose winningMotor action.
        largest = []
        for i, v in enumerate( motorScore ):
            if sorted( motorScore, reverse=True )[ 0 ] == v:
                largest.append( i )                                         # 0 = UP, 1 = STILL, 2 = DOWN
        winningMotor = random.choice( largest )

        # Use winningMotor to support synapses leading to this one and inhibit those not.
#        for cell in predToMotor.sparse:
#            for i in range( self.motorDimensions ):
#                if i == winningMotor:
#                    self.motorSynapse[ ( cell * self.motorDimensions ) + i ] += 0.001
#                else:
#                    self.motorSynapse[ ( cell * self.motorDimensions ) + i ] -= 0.001

        if self.manualInput != -1:
            # If motor suggestion, manualInput, equals winningMotor then send a small reward.
            if winningMotor == self.manualInput:
                self.Hippocampus( 1.0 )
            # If not send a small punishment.
            else:
                self.Hippocampus( -1.0 )

        # Return winning motor function.
        return winningMotor

        #----------------------------------------------------------------------

# Create play agents
leftAgent = Agent( 'Left' )
rightAgent = Agent( 'Right' )

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

def on_press(key):
    if key.char   == ( 'w' ):
        leftAgent.SendSuggest( 0 )
    elif key.char == ( 's' ):
        leftAgent.SendSuggest( 2 )
    elif key == Key.escape:
        sys.exit( "Quit program, escape key pressed." )


def on_release(key):
    leftAgent.SendSuggest( -1 )

listener = keyboard.Listener( on_press = on_press, on_release = on_release )
listener.start()

while True:
# Main game loop

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
        rightAgent.Hippocampus( -1.0 )
        leftAgent.Reset()
        rightAgent.Reset()

    elif ball.xcor() < -350:
        # Ball falls off left side of screen. B gets a point.
        score_b += 1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a, score_b), align="center", font=("Courier", 24, "normal"))
        ball.goto(0, 0)
        ball.dx *= -1
        ball.dy *= random.choice( [ -1, 1 ] )
        leftAgent.Hippocampus( -1.0 )
        leftAgent.Reset()
        rightAgent.Reset()

    # Paddle and ball collisions
    if ball.xcor() < -340 and ball.ycor() < paddle_a.ycor() + 50 and ball.ycor() > paddle_a.ycor() - 50:
        # Ball hits paddle A
        ball.dx *= -1
        ball.goto( -340, ball.ycor() )
        leftAgent.Hippocampus( 1.0 )
        leftAgent.Reset()
        rightAgent.Reset()

    elif ball.xcor() > 340 and ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() > paddle_b.ycor() - 50:
        # Ball hits paddle B
        ball.dx *= -1
        ball.goto( 340, ball.ycor() )
        rightAgent.Hippocampus( 1.0 )
        leftAgent.Reset()
        rightAgent.Reset()

    # Run each agents learning algorithm and produce movement.
    leftMove = leftAgent.Brain( paddle_a.ycor(), ball.xcor(), paddle_a.ycor() - ball.ycor(), ball.dx, ball.dy )
    rightMove = rightAgent.Brain( paddle_b.ycor(), ball.xcor(), paddle_b.ycor() - ball.ycor(), ball.dx, ball.dy )

    if leftMove == 0:
        paddle_a_up()
    elif leftMove == 2:
        paddle_a_down()

    if rightMove == 0:
        paddle_b_up()
    elif rightMove == 2:
        paddle_b_down()
