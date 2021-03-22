# Simple Pong game combined with Nupic AI, in Python 3
# By Anand Chotai

#import csv
#import datetime
import numpy
import random

from agent_ball_proto import BallAgent

# Dimensions of screen.
screenWidth  = 800
screenHeight = 600

# Set up Turtle screen stuff----------------------------------------------------
import turtle

wn = turtle.Screen()
wn.title( "Pong" )
wn.bgcolor( "black" )
wn.setup( width = screenWidth, height = screenHeight )
wn.tracer( 0 )

# Set up ball
ballHeight  = 1                            # Stretch of ball
ballWidth   = 1
ballSpeed = 20

ball = turtle.Turtle()
ball.speed( 0 )
ball.shape( "square" )
ball.color( "white" )
ball.shapesize( stretch_wid = ballHeight, stretch_len = ballWidth )
ball.penup()
ball.goto( 0, 0 )
ball.dx = random.choice( [ -ballSpeed, ballSpeed ] )
ball.dy = random.choice( [ -ballSpeed, ballSpeed ] )

# Paddles
paddleHeight  = 5                            # Stretch of paddles
paddleWidth   = 1
paddleAMove = True
paddleBMove = True

paddle_a = turtle.Turtle()
paddle_a.speed( 0 )
paddle_a.shape( "square" )
paddle_a.color( "white" )
paddle_a.shapesize( stretch_wid = paddleHeight, stretch_len = paddleWidth )
paddle_a.penup()
paddle_a.goto( -350, -180 )

paddle_b = turtle.Turtle()
paddle_b.speed( 0 )
paddle_b.shape( "square" )
paddle_b.color( "white" )
paddle_b.shapesize( stretch_wid = paddleHeight, stretch_len = paddleWidth )
paddle_b.penup()
paddle_b.goto( 350, 180 )

# Create play agents------------------------------------------------------------
ballAgent = BallAgent( 'BallAgent', screenHeight, screenWidth, ballHeight, ballWidth, paddleHeight, paddleWidth )

# Set up pred locations.
drawLength = ballAgent.maxPredLocations
placeCellsDraw = []
for i in range( drawLength ):
    currPlaceDraw = turtle.Turtle( )
    currPlaceDraw.speed( 0 )
    currPlaceDraw.shape( "circle" )
    currPlaceDraw.color( "red" )
    currPlaceDraw.shapesize( stretch_wid = 0.5, stretch_len = 0.5 )
    currPlaceDraw.penup( )
    currPlaceDraw.goto( 0, 0 )
    placeCellsDraw.append( currPlaceDraw )

# Functions---------------------------------------------------------------------
def Within ( value, minimum, maximum, equality ):
# Checks if value is <= maximum and >= minimum.

    if equality:
        if value <= maximum and value >= minimum:
            return True
        else:
            return False
    else:
        if value < maximum and value > minimum:
            return True
        else:
            return False

while True:
# Main game loop----------------------------------------------------------------

    wn.update()         # Screen update

    # Move the ball
    ball.setx( ball.xcor() + ball.dx )
    ball.sety( ball.ycor() + ball.dy )

    # Border checking top and bottom.
    if ball.ycor() > int( screenHeight / 2 ) - ( ballHeight * 10 ):
        ball.sety( int( screenHeight / 2 ) - ( ballHeight * 10 ) )
        ball.dy *= -1

    elif ball.ycor() < -int( screenHeight / 2 ) + ( ballHeight * 10 ):
        ball.sety( -int( screenHeight / 2 ) + ( ballHeight * 10 ) )
        ball.dy *= -1

    # Border checking left and right.
    if ball.xcor() > int( screenWidth / 2 ) - ( ballWidth * 10 ):
        ball.goto(0, 0)
#        ball.setx( int( screenWidth / 2 ) - ( ballWidth * 10 ) )
        ball.dx *= -1
        ball.dy *= random.choice( [ -1, 1 ] )
        paddleBMove = True

    elif ball.xcor() < -int( screenWidth / 2 ) + ( ballWidth * 10 ):
        ball.goto(0, 0)
#        ball.setx( -int( screenWidth / 2 ) + ( ballWidth * 10 ) )
        ball.dx *= -1
        ball.dy *= random.choice( [ -1, 1 ] )
        paddleAMove = True

    # Paddle and ball collisions.
    if ball.xcor() < -340 and ball.ycor() < paddle_a.ycor() + 50 and ball.ycor() > paddle_a.ycor() - 50:
        # Ball hits paddle A
        ball.dx *= -1
        ball.goto( -340, ball.ycor() )
#        paddleAMove = False

    elif ball.xcor() > 340 and ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() > paddle_b.ycor() - 50:
        # Ball hits paddle B
        ball.dx *= -1
        ball.goto( 340, ball.ycor() )
#        paddleBMove = False

    # Run each agents learning algorithm and produce predictions.
    paddleMove = ballAgent.Brain( ball.xcor(), ball.ycor(), paddle_a.ycor(), paddle_b.ycor() )

    # Move paddles.
    if paddleMove[ 0 ] == 0 and paddleAMove:
        paddle_a.sety( paddle_a.ycor() + 20 )
        if paddle_a.ycor() >= screenHeight / 2:
            paddle_a.sety( screenHeight / 2 )
    elif paddleMove[ 0 ] == 2 and paddleAMove:
        paddle_a.sety( paddle_a.ycor() - 20 )
        if paddle_a.ycor() <= -screenHeight / 2:
            paddle_a.sety( -screenHeight / 2 )
    if paddleMove[ 1 ] == 0 and paddleBMove:
        paddle_b.sety( paddle_b.ycor() + 20 )
        if paddle_b.ycor() >= screenHeight / 2:
            paddle_b.sety( screenHeight / 2 )
    elif paddleMove[ 1 ] == 2 and paddleBMove:
        paddle_b.sety( paddle_b.ycor() - 20 )
        if paddle_b.ycor() <= -screenHeight / 2:
            paddle_b.sety( -screenHeight / 2 )

    # Draw predictions for ball agent.
    for i in range( len( ballAgent.predPositions ) ):
        if Within( ballAgent.predPositions[ i ][ 0 ], -screenWidth / 2, screenWidth / 2, True ) and Within( ballAgent.predPositions[ i ][ 1 ], -screenHeight / 2, screenHeight / 2, True ):
            placeCellsDraw[ i ].setx( ballAgent.predPositions[ i ][ 0 ] )
            placeCellsDraw[ i ].sety( ballAgent.predPositions[ i ][ 1 ] )
        else:
            placeCellsDraw[ i ].setx( 0 )
            placeCellsDraw[ i ].sety( 0 )
