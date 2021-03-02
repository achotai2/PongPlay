# Simple Pong game combined with Nupic AI, in Python 3
# By Anand Chotai

#import csv
#import datetime
import numpy
import random

from agent_ball import BallAgent

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

# Create play agents------------------------------------------------------------
ballAgent = BallAgent( 'BallAgent', screenHeight, screenWidth, ballHeight, ballWidth, paddleHeight, paddleWidth )

# Set up pred locations.
drawLength = ballAgent.maxMemoryDist
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

# Set up agent attention square.
#attentSqDraw = turtle.Turtle( )
#attentSqDraw.speed( 0 )
#attentSqDraw.shape( "square" )
#attentSqDraw.color( "red" )
#attentSqDraw.shapesize( stretch_wid = ballAgent.localDimX / 20, stretch_len = ballAgent.localDimY / 20 )
#attentSqDraw.penup( )
#attentSqDraw.goto( ballAgent.centerX, ballAgent.centerY )

# Functions---------------------------------------------------------------------
def Within ( value, minimum, maximum ):
# Checks if value is <= maximum and >= minimum.

    if value <= maximum and value >= minimum:
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
#        ball.goto(0, 0)
        ball.setx( int( screenWidth / 2 ) - ( ballWidth * 10 ) )
        ball.dx *= -1
#        ball.dy *= random.choice( [ -1, 1 ] )

    elif ball.xcor() < -int( screenWidth / 2 ) + ( ballWidth * 10 ):
#        ball.goto(0, 0)
        ball.setx( -int( screenWidth / 2 ) + ( ballWidth * 10 ) )
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

    # Run each agents learning algorithm and produce predictions.
    ballAgent.Brain( ball.xcor(), ball.ycor() )

    # Draw attention square for ball agent.
#    attentSqDraw.setx( ballAgent.centerX )
#    attentSqDraw.sety( ballAgent.centerY )

    # Draw predictions for ball agent.
    for i in range( drawLength ):
        if i <= len( ballAgent.predPositions ) - 1:
            placeCellsDraw[ i ].setx( ballAgent.predPositions[ i ][ 0 ] )
            placeCellsDraw[ i ].sety( ballAgent.predPositions[ i ][ 1 ] )
        else:
            placeCellsDraw[ i ].setx( 0 )
            placeCellsDraw[ i ].sety( 0 )
