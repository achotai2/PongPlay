# Simple Pong game combined with Nupic AI, in Python 3
# By Anand Chotai

import csv
import datetime
import turtle
import numpy
import yaml
import random
import atexit

from agent_main import Agent
from logs_yo import Logging

screenHeight   = 600          # Used in setting up screen and encoders and agents
screenWidth    = 800
senseResX      = 150
senseResY      = 150
ballWidth      = 20
ballHeight     = 20
paddleWidth    = 20
paddleHeight   = 200
xSpeed         = -10           # Speed for ball
ySpeed         = 0

# Create play agents
leftAgent  = Agent( 'Left', senseResX, senseResY, screenHeight, screenWidth, ballWidth, ballHeight, paddleWidth, paddleHeight )
#rightAgent = Agent( 'Right', senseResX, senseResY, screenHeight, screenWidth, ballWidth, ballHeight, paddleWidth, paddleHeight )

leftAgentReflective  = False
#rightAgentReflective = False

#logCompile = Logging( [ leftAgent.ID, rightAgent.ID ] )
logCompile = Logging( [ leftAgent.ID ] )

leftScore  = 0
rightScore = 0

# Set up Turtle screen
wn = turtle.Screen()
wn.title( "Pong" )
wn.bgcolor( "black" )
wn.setup( width = screenWidth, height = screenHeight )
wn.tracer( 0 )

# Paddle A
paddle_a = turtle.Turtle()
paddle_a.speed( 0 )
paddle_a.shape( "square" )
paddle_a.color( "white" )
paddle_a.shapesize( stretch_wid = paddleHeight / 20, stretch_len = paddleWidth / 20 )
paddle_a.penup()
paddle_a.goto( -350, 0 )

# Paddle B
#paddle_b = turtle.Turtle()
#paddle_b.speed( 0 )
#paddle_b.shape( "square" )
#paddle_b.color( "white" )
#paddle_b.shapesize( stretch_wid = paddleHeight / 20, stretch_len = paddleWidth / 20 )
#paddle_b.penup()
#paddle_b.goto( 350, 0 )

# Ball
ball = turtle.Turtle()
ball.speed( 0 )
ball.shape( "square" )
ball.color( "white" )
ball.shapesize( stretch_wid = ballWidth / 20, stretch_len = ballHeight / 20 )
ball.penup()
ball.goto( 0, 0 )
ball.dx = xSpeed
ball.dy = ySpeed

# Sense Organ A.
agent1SenseX, agent1SenseY = leftAgent.ReturnSenseOrganLocation()
senseOrgan1 = turtle.Turtle( )
senseOrgan1.speed( 0 )
senseOrgan1.shape( "square" )
senseOrgan1.color( "blue" )
senseOrgan1.shapesize( stretch_wid = senseResX / 20, stretch_len = senseResY / 20 )
senseOrgan1.penup()
senseOrgan1.goto( agent1SenseX, agent1SenseY )

# Sense Organ B.
#agent2SenseX, agent2SenseY = rightAgent.ReturnSenseOrganLocation()
#senseOrgan2 = turtle.Turtle( )
#senseOrgan2.speed( 0 )
#senseOrgan2.shape( "square" )
#senseOrgan2.color( "red" )
#senseOrgan2.shapesize( stretch_wid = senseResX / 20, stretch_len = senseResY / 20 )
#senseOrgan2.penup()
#senseOrgan2.goto( agent2SenseX, agent2SenseY )

# Pen to display score
pen = turtle.Turtle()
pen.speed( 0 )
pen.shape( "square" )
pen.color( "white" )
pen.penup()
pen.hideturtle()
pen.goto( 0, 260 )
pen.write(
    "Left Agent: {}% on {} events".format( 0, 0 ),
    align = "center", font = ( "Courier", 24, "normal" )
)
pen.goto( 0, 230 )
pen.write(
    "Right Agent: {}% on {} events".format( 0, 0 ),
    align = "center", font = ( "Courier", 24, "normal" )
)

# Functions
def ReDrawScore():
    pen.clear()
    pen.goto( 0, 260 )
    pen.write(
        "Left Agent: {}".format( leftScore ),
        align = "center", font = ( "Courier", 24, "normal" )
    )
    pen.goto( 0, 230 )
    pen.write(
        "Right Agent: {}".format( rightScore ),
        align = "center", font = ( "Courier", 24, "normal" )
    )

def paddle_a_up():
    y = paddle_a.ycor()
    if y < 290 - 100:
        y += 20
        paddle_a.sety( y )

def paddle_a_down():
    y = paddle_a.ycor()
    if y > -290 + 100:
        y -= 20
        paddle_a.sety( y )

#def paddle_b_up():
#    y = paddle_b.ycor()
#    if y < 290 - 100:
#        y += 20
#        paddle_b.sety( y )

#def paddle_b_down():
#    y = paddle_b.ycor()
#    if y > -290 + 100:
#        y -= 20
#        paddle_b.sety( y )

def ResetBall():
# Reset ball location and speed.

    ball.goto( 0, random.randrange( -( screenHeight * 0.5 ) + 20, ( screenHeight * 0.5 ) - 20 ) )
    ball.dy *= random.choice( [ -1, 1 ] )
    ball.dx = -10

def exit_handler():
# Upon program exit collects data for Cell-Report log file, and produces the final plot.

#    logCompile.WhenExit( [ leftAgent, rightAgent ] )
    logCompile.WhenExit( [ leftAgent ] )

atexit.register( exit_handler )

while True:
# Main game loop

    wn.update()         # Screen update

    logCompile.AddToTimeStep()

#    if not leftAgentReflective and not rightAgentReflective:
    if not leftAgentReflective:
        leftFeeling = 0.0

        # Move the ball
        ball.setx( ball.xcor() + ball.dx )
        ball.sety( ball.ycor() + ball.dy )

        leftFeeling  = 0.0
#        rightFeeling = 0.0

        # Border checking top and bottom.
        if ball.ycor() > 290:
            ball.sety( 290 )
            ball.dy *= -1

        elif ball.ycor() < -290:
            ball.sety( -290 )
            ball.dy *= -1

        # Border checking left and right.
#        if ball.xcor() > 350:
#            # Ball falls off right side of screen. A gets a point.
#            leftScore += 1
#            ReDrawScore()

#            ball.goto( 0, 0 )
#            ball.dx *= -1
#            ball.dy *= random.choice( [ -1, 1 ] )

#            rightFeeling = -1.0
#            rightAgentReflective = True

        elif ball.xcor() < -350:
            # Ball falls off left side of screen. B gets a point.
#            rightScore += 1
#            ReDrawScore()

#            ball.goto( 0, 0 )
#            ball.dx *= -1
#            ball.dy *= random.choice( [ -1, 1 ] )

            ResetBall()

            leftFeeling  = -1.0
            leftAgentReflective = True

        # Paddle and ball collisions
        if ball.xcor() < -340 and ball.ycor() < paddle_a.ycor() + ( paddleHeight / 2 ) and ball.ycor() > paddle_a.ycor() - ( paddleHeight / 2 ):
            # Ball hits paddle A
#            ball.dx *= -1
#            ball.goto( -340, ball.ycor() )

            ResetBall()

            leftFeeling  = 1.0
            leftAgentReflective = True

#        elif ball.xcor() > 340 and ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() > paddle_b.ycor() - 50:
#            # Ball hits paddle B
#            ball.dx *= -1
#            ball.goto( 340, ball.ycor() )

#            rightFeeling = 1.0
#            rightAgentReflective = True

        # Run each agents learning algorithm and produce movement.
#        leftMove  = leftAgent.Brain( paddle_a.xcor(), paddle_a.ycor(), paddle_b.xcor(), paddle_b.ycor(), ball.xcor(), ball.ycor(), ball.dx, ball.dy, leftFeeling )
#        rightMove = rightAgent.Brain( paddle_a.xcor(), paddle_a.ycor(), paddle_b.xcor(), paddle_b.ycor(), ball.xcor(), ball.ycor(), ball.dx, ball.dy, rightFeeling )
        leftMove  = leftAgent.Brain( paddle_a.xcor(), paddle_a.ycor(), ball.xcor(), ball.ycor(), ball.dx, ball.dy, leftFeeling )

    else:
       if leftAgentReflective:
            leftAgentReflective = leftAgent.Reflect( leftFeeling )
#        if rightAgentReflective:
#            rightAgentReflective = rightAgent.Reflect()

#    logCompile.AccumulateReportData( [ leftAgent, rightAgent ], [ paddle_a.xcor(), paddle_a.ycor(), paddle_b.xcor(), paddle_b.ycor(), ball.xcor(), ball.ycor(), ball.dx, ball.dy ] )
    logCompile.AccumulateReportData( [ leftAgent ], [ paddle_a.xcor(), paddle_a.ycor(), ball.xcor(), ball.ycor(), ball.dx, ball.dy ] )

    agent1SenseX, agent1SenseY = leftAgent.ReturnSenseOrganLocation()
    senseOrgan1.goto( agent1SenseX, agent1SenseY )
#    agent2SenseX, agent2SenseY = rightAgent.ReturnSenseOrganLocation()
#    senseOrgan2.goto( agent2SenseX, agent2SenseY )

    leftMove = 1

    # Move agents.
    if leftMove == 0:
        paddle_a_up()
    elif leftMove == 2:
        paddle_a_down()

#    if rightMove == 0:
#        paddle_b_up()
#    elif rightMove == 2:
#        paddle_b_down()

#    logCompile.WriteDataToFiles( [ leftAgent, rightAgent ] )
    logCompile.WriteDataToFiles( [ leftAgent ] )
