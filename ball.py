# Simple Pong game combined with Nupic AI, in Python 3
# By Anand Chotai

import csv
import datetime
import numpy
import yaml
import random

from agent_ball import BallAgent

# Dimensions of screen.
screenWidth  = 800
screenHeight = 600

# Create play agents
ballAgent = BallAgent( 'BallAgent', screenHeight, screenWidth )

# Set up Turtle screen
import turtle

wn = turtle.Screen()
wn.title( "Pong" )
wn.bgcolor( "black" )
wn.setup( width = screenWidth, height = screenHeight )
wn.tracer( 0 )

# Set up ball
xSpeed = random.choice( [ -20, 20 ] )       # Speed for ball.
ySpeed = random.choice( [ -20, 20 ] )

ballHeight  = 1                            # Stretch of ball
ballWidth   = 1

ball = turtle.Turtle()
ball.speed( 0 )
ball.shape( "square" )
ball.color( "white" )
ball.shapesize( stretch_wid = ballHeight, stretch_len = ballWidth )
ball.penup()
ball.goto( 0, 0 )
ball.dx = xSpeed
ball.dy = ySpeed

# Paddles
paddleHeight  = 5                            # Stretch of paddles
paddleWidth   = 1

paddle_a = turtle.Turtle()
paddle_a.speed( 0 )
paddle_a.shape( "square" )
paddle_a.color( "white" )
paddle_a.shapesize( stretch_wid = paddleHeight, stretch_len = paddleWidth )
paddle_a.penup()
paddle_a.goto( -350, -100 )

paddle_b = turtle.Turtle()
paddle_b.speed( 0 )
paddle_b.shape( "square" )
paddle_b.color( "white" )
paddle_b.shapesize( stretch_wid = paddleHeight, stretch_len = paddleWidth )
paddle_b.penup()
paddle_b.goto( 350, 240 )

# Set up pred locations.
placeCellsDraw = []
for i in range( 10 ):
    currPlaceDraw = turtle.Turtle( )
    currPlaceDraw.speed( 0 )
    currPlaceDraw.shape( "circle" )
    currPlaceDraw.color( "red" )
    currPlaceDraw.shapesize( stretch_wid = 0.5, stretch_len = 0.5 )
    currPlaceDraw.penup( )
    currPlaceDraw.goto( 0, 0 )
    placeCellsDraw.append( currPlaceDraw )

def CreateBitRep():
# Create a bit representation of entire screen. This is a list that stores the indices of ON bits.

    screenBitRep = []

    # Ball bits.
    for y in range( -ballHeight * 10, ballHeight * 10 ):
        for x in range( -ballWidth * 10, ballWidth * 10 ):
            screenBitRep.append( int( ball.xcor() + ( screenWidth / 2 ) + x + ( ( ball.ycor() + ( screenHeight / 2 ) + y ) * screenWidth ) ) )

    # Paddle A bits.
    for y in range( -paddleHeight * 10, paddleHeight * 10 ):
        for x in range( -paddleWidth * 10, paddleWidth * 10 ):
            screenBitRep.append( int( paddle_a.xcor() + ( screenWidth / 2 ) + x + ( ( paddle_a.ycor() + ( screenHeight / 2 ) + y ) * screenWidth ) ) )

    # Paddle B bits.
    for y in range( -paddleHeight * 10, paddleHeight * 10 ):
        for x in range( -paddleWidth * 10, paddleWidth * 10 ):
            screenBitRep.append( int( paddle_b.xcor() + ( screenWidth / 2 ) + x + ( ( paddle_b.ycor() + ( screenHeight / 2 ) + y ) * screenWidth ) ) )

    return numpy.unique( screenBitRep )

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
#        ball.goto(0, 0)
        ball.setx( 350 )
        ball.dx *= -1
#        ball.dy *= random.choice( [ -1, 1 ] )

    elif ball.xcor() < -350:
#        ball.goto(0, 0)
        ball.setx( -350 )
        ball.dx *= -1
#        ball.dy *= random.choice( [ -1, 1 ] )

    # Paddle and ball collisions
#    if ball.xcor() < -340 and ball.ycor() < paddle_a.ycor() + 50 and ball.ycor() > paddle_a.ycor() - 50:
        # Ball hits paddle A
#        ball.dx *= -1
#        ball.goto( -340, ball.ycor() )

#    elif ball.xcor() > 340 and ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() > paddle_b.ycor() - 50:
        # Ball hits paddle B
#        ball.dx *= -1
#        ball.goto( 340, ball.ycor() )

    # Run each agents learning algorithm and produce movement.
    CreateBitRep()
    predPositions = ballAgent.Brain( ball.xcor(), ball.ycor() )

    for i in range( 10 ):
        if i <= len( predPositions ) - 1:
            placeCellsDraw[ i ].setx( predPositions[ i ][ 0 ] )
            placeCellsDraw[ i ].sety( predPositions[ i ][ 1 ] )
