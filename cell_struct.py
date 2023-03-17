from bisect import bisect_left
from random import uniform, choice, sample, randrange, shuffle
from collections import Counter
from useful_functions import BinarySearch, NoRepeatInsort, IndexIfItsIn, FastIntersect, GenerateUnitySDR, NumStandardDeviations, CalculateDistanceScore, DelIfIn
import math

class Cell:

    def __init__( self, colID, vectorMemoryDict ):
    # Create a new feature level cell with synapses to OCells.

        self.vectorMemoryDict = vectorMemoryDict

        # FCell state variables.
        self.active     = False
        self.lastActive = False
        self.predicted  = False
        self.winner     = False
        self.lastWinner = False

        self.column     = colID

        self.asIncident = []            # Keeps track of what segments this cell is incident on.
        self.asTerminal = []

        # For data collection.
        self.activeCount = 0
        self.states      = []
        self.statesCount = []

#    def __repr__( self ):
#    # Returns string properties of the FCell.
#
#        toPrint = []
#        for index in range( len( self.OCellConnections ) ):
#            toPrint.append( ( self.OCellConnections[ index ], self.OCellPermanences[ index ] ) )
#
#        return ( "< ( Connected OCells, Permanence ): %s >"
#            % toPrint )

    def ReceiveStateData( self, stateNumber ):
    # Receive the state number and record it along with the count times, if this cell is active.

        if self.active:
            self.activeCount += 1

            stateIndex = bisect_left( self.states, stateNumber )
            if stateIndex != len( self.states ) and self.states[ stateIndex ] == stateNumber:
                self.statesCount[ stateIndex ] += 1
            else:
                self.states.insert( stateIndex, stateNumber )
                self.statesCount.insert( stateIndex, 1 )

    def ReturnStateInformation( self ):
    # Return the state information and count.

        toReturn = []
        toReturn.append( self.activeCount )
        for i, s in enumerate( self.states ):
            toReturn.append( ( s, self.statesCount[ i ] ) )

        return toReturn

    def IncidentToThisSeg( self, segIndex ):
    # Add reference that this cell is on this segment.

        lengthBefore = len( self.asIncident )

        NoRepeatInsort( self.asIncident, segIndex )

        if len( self.asIncident ) == lengthBefore:
            print( "IncidentToThisSeg(): Tried to add reference to segment, but already exists." )
            exit()

        if len( self.asIncident ) > self.vectorMemoryDict[ "maxIncidentOnCell" ]:
            return True
        else:
            return False

    def TerminalToThisSeg( self, segIndex ):
    # Add reference that this cell is on this segment.

        lengthBefore = len( self.asTerminal )

        NoRepeatInsort( self.asTerminal, segIndex )

        if len( self.asTerminal ) == lengthBefore:
            print( "TerminalToThisSeg(): Tried to add reference to segment, but already exists." )
            exit()

    def DeleteIncidentSegmentReference( self, segIndex ):
    # Checks if this cell is incident to this segment, if so then delete it. Also lower all segment references by one.

        DelIfIn( self.asIncident, segIndex )

    def DeleteTerminalSegmentReference( self, segIndex ):
    # Checks if this cell is incident to this segment, if so then delete it. Also lower all segment references by one.

        DelIfIn( self.asTerminal, segIndex )

    def ReturnColumn( self ):
    # Return my column.

        return self.column

    def ReturnIncidentOn( self ):
    # Returns this cells column, and a list of segments this cell is incident on.

        return self.asIncident.copy()

    def ReturnTerminalOn( self ):

        return self.asTerminal.copy()

    def ConnectionToSegment( segIndex ):
    # Returns true if this cell is incident on segment segIndex, otherwise returns False.

        return BinarySearch( self.asIncident, segIndex )

    def SetAsWinner( self ):
    # Set as winner.

        self.winner = True

    def IsWinner( self ):
    # Return True if it is winner.

        return self.winner

    def MakeActive( self ):
    # Make cell active.

        self.active = True
