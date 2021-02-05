# Simple Pong game combined with Nupic AI, in Python 3
# By Anand Chotai

import csv
import datetime
import numpy
import yaml
import random
import time
from pynput import keyboard

from agent_main import Agent
from agent_proto import AgentProto

screenHeight = 600          # Used in setting up screen and encoders and agents
screenWidth = 800

# Create play agents
leftAgent = Agent( 'Left', screenHeight, screenWidth )
rightAgent = AgentProto( 'Right', screenHeight, screenWidth )

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

# Functions
def ReDrawScore():
    pen.clear()
    pen.goto(0, 260)
    pen.write(
        "Left Agent: {}% on {} events".format(
            round( leftAgent.percentSuccess, 1 ),
            leftAgent.numEvents,
        ),
        align = "center", font = ("Courier", 24, "normal")
    )
    pen.goto(0, 230)
    pen.write(
        "Right Agent: {}% on {} events".format(
            round( rightAgent.percentSuccess, 1 ),
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
    elif key.char == ( 'w' ):
        leftAgent.SendSuggest( 0 )
        print("w pressed")
    elif key.char == ( 's' ):
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
        rightAgent.Hippocampus( p )
        leftAgent.Clear()
        rightAgent.Clear()

        rightAgent.UpdateScore( False )
        ReDrawScore()

        ball.goto(0, 0)
        ball.dx *= -1
        ball.dy *= random.choice( [ -1, 1 ] )

    elif ball.xcor() < -350:
        # Ball falls off left side of screen. B gets a point.
        p = -( numpy.arctan( ( numpy.absolute( paddle_a.ycor() - ball.ycor() ) - ( screenHeight / 2 ) ) / 10 ) / numpy.pi ) - 0.5
        leftAgent.Hippocampus( p )
        leftAgent.Clear()
        rightAgent.Clear()

        leftAgent.UpdateScore( False )
        ReDrawScore()

        ball.goto(0, 0)
        ball.dx *= -1
        ball.dy *= random.choice( [ -1, 1 ] )

    # Paddle and ball collisions
    if ball.xcor() < -340 and ball.ycor() < paddle_a.ycor() + 50 and ball.ycor() > paddle_a.ycor() - 50:
        # Ball hits paddle A
        leftAgent.Hippocampus( 1.0 )
        leftAgent.Clear()
        rightAgent.Clear()

        leftAgent.UpdateScore( True )
        ReDrawScore()

        ball.dx *= -1
        ball.goto( -340, ball.ycor() )

    elif ball.xcor() > 340 and ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() > paddle_b.ycor() - 50:
        # Ball hits paddle B
        rightAgent.Hippocampus( 1.0 )
        leftAgent.Clear()
        rightAgent.Clear()

        rightAgent.UpdateScore( True )
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
