# Simple Pong game combined with Nupic AI, in Python 3
# By Anand Chotai

import csv
import datetime
import numpy
import yaml
import random

from agent_ball import BallAgent

screenHeight = 600          # Used in setting up screen and encoders and agents
screenWidth = 800

# Create play agents
ballAgent = BallAgent( 'BallAgent', screenHeight, screenWidth )

xSpeed = random.choice( [ -20, 20 ] )                   # Speed for ball
ySpeed = random.choice( [ -20, 20 ] )

# Set up Turtle screen
import turtle

wn = turtle.Screen()
wn.title("Pong")
wn.bgcolor("black")
wn.setup(width=screenWidth, height=screenHeight)
wn.tracer(0)

# Set up ball
ball = turtle.Turtle()
ball.speed(0)
ball.shape("square")
ball.color("white")
ball.penup()
ball.goto(0, 0)
ball.dx = xSpeed
ball.dy = ySpeed

# Set up pred locations.
placeCellsDraw = []
for i in range( 10 ):
    currPlaceDraw = turtle.Turtle( )
    currPlaceDraw.speed( 0 )
    currPlaceDraw.shape( "circle" )
    currPlaceDraw.color( "red" )
    currPlaceDraw.shapesize( stretch_wid=0.5, stretch_len=0.5 )
    currPlaceDraw.penup( )
    currPlaceDraw.goto( 0, 0 )
    placeCellsDraw.append( currPlaceDraw )

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
        ball.setx( 350 )
        ball.dx *= -1

    elif ball.xcor() < -350:
        ball.setx( -350 )
        ball.dx *= -1

    # Run each agents learning algorithm and produce movement.
    predPositions = ballAgent.Brain( ball.xcor(), ball.ycor() )

    for i in range( 10 ):
        if i <= len( predPositions ) - 1:
            placeCellsDraw[ i ].setx( predPositions[ i ][ 0 ] )
            placeCellsDraw[ i ].sety( predPositions[ i ][ 1 ] )
