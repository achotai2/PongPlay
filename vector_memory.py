from random import sample
from bisect import bisect_left
from cell_and_synapse import FCell, OCell
import numpy as np
from time import time

def BinarySearch( list, val ):
# Search a sorted list: return False if val not in list, and True if it is.

    i = bisect_left( list, val )
    if i != len( list ) and list[ i ] == val:
        return True
    else:
        return False

class VectorMemory:

    def __init__( self, columnDimensions, cellsPerColumn, numObjectCells, activationThreshold, initialPermanence,
        connectedPermanence, permanenceIncrement, permanenceDecrement, initialPosVariance, OCellActivation,
        maxNewSynapseCount, shiftMultiplier, initialFlexibility, maxSegmentsPerCell, maxSynapsesPerSegment ):
    # columnDimensions      = Dimensions of the column space.
    # cellsPerColumn        = Number of cells per column.
    # numObjectCells        = Number of cells in the Object level.
    # activationThreshold   = If the number of active connected synapses on a segment is at least this threshold,
    #                         the segment is said to be active.
    # initialPermanence     = Initial permanence of a new synapse.
    # connectedPermanence   = If the permanence value for a synapse is greater than this value, it is said to be connected.
    # permanenceIncrement   = Amount by which permanences of synapses are incremented during learning.
    # permanenceDecrement   = Amount by which permanences of synapses are decremented during learning.
    # initialPosVariance    = Amount of range vector positions are valid in.
    # OCellActivation       = Number of active Ocells to be considered a valid object represention.
    # maxNewSynapseCount    = The maximum number of synapses added to a segment during learning.
    # shiftMultiplier       = The value to multiply vector shifts by.
    # initialFlexibility    = The initialFlexibility value.
    # maxSegmentsPerCell    = The maximum number of segments per cell.
    # maxSynapsesPerSegment = The maximum number of synapses per segment.

        self.columnDimensions      = columnDimensions
        self.FCellsPerColumn       = cellsPerColumn
        self.numObjectCells        = numObjectCells
        self.activationThreshold   = activationThreshold
        self.initialPermanence     = initialPermanence
        self.connectedPermanence   = connectedPermanence
        self.permanenceIncrement   = permanenceIncrement
        self.permanenceDecrement   = permanenceDecrement
        self.initialPosVariance    = initialPosVariance
        self.OCellActivation       = OCellActivation
        self.maxNewSynapseCount    = maxNewSynapseCount
        self.shiftMultiplier       = shiftMultiplier
        self.flexibility           = initialFlexibility
        self.maxSegmentsPerCell    = maxSegmentsPerCell
        self.maxSynapsesPerSegment = maxSynapsesPerSegment

        # --------Create all the cells in the network.---------
        # Create cells in feature layer.
        self.FCells = []
        for i in range( columnDimensions * cellsPerColumn ):
            self.FCells.append( FCell( i ) )
        self.activeFCells     = []
        self.lastActiveFCells = []
        self.predictiveFCells = []
        self.primedFCells     = []

        # Create cells in object layer.
        self.OCells = []
        for o in range( numObjectCells ):
            self.OCells.append( OCell( o ) )
            # Create a random set of active object cells to start.
        self.activeOCells = sorted( sample( range( numObjectCells ), OCellActivation ) )

    def SelectWinnerCell( self, burstingColumn ):
    # Selects one winner cell from the bursting column, choosing the one with the least segments.

        minSegs    = -1
        minSegIndx = 0

        for cell in range( burstingColumn * self.FCellsPerColumn, ( burstingColumn * self.FCellsPerColumn ) + self.FCellsPerColumn ):
            if minSegs == -1 or len( self.FCells[ cell ].segments ) < minSegs:
                minSegs    = len( self.FCells[ cell ].segments )
                minSegIndx = cell

        return minSegIndx

    def NewFeature( self, burstingCols, vector ):
    # Create a new feature.

        winnerCells = []

        for col in burstingCols:
            # Select winner cells from all bursting columns.
            winnerCells.append( self.SelectWinnerCell( col ) )

        # Create synapses from all past primed FCells to these winnerCells.
        if len( self.lastActiveFCells ) > 0:
            lastCell = self.lastActiveFCells[ 0 ]
            # Go through all the cells of all the segments in every lastActiveFCell
            if len( self.FCells[ lastCell ].segments ) > 0:
                for segment in self.FCells[ lastCell ].segments:

                    # Calculate vectors that go from this other cell, through the lastActive cell, to the winnerCell, and vise-versa.
                    oldToNewVector = [ 0 ] * len( vector )
                    newToOldVector = [ 0 ] * len( vector )
                    for i in range( len( vector ) ):
                        oldToNewVector[ i ] = vector[ i ] - segment.vector[ i ][ 0 ]
                        newToOldVector[ i ] = segment.vector[ i ][ 0 ] - vector[ i ]

                    # Create a new segment from every cell on every segment of lastActiveFCells' cells.
                    for endCell in segment.synapses:
                        self.FCells[ endCell ].CreateSegment( winnerCells, self.initialPermanence, oldToNewVector, self.initialPosVariance )

                    # Create a segment from this winnerCell to this lastActive segment cells.
                    for winCell in winnerCells:
                        self.FCells[ winCell ].CreateSegment( segment.synapses, self.initialPermanence, newToOldVector, self.initialPosVariance )

            # Create for last active cells a new segment to these winner cells.
            # Create winner cells from lastActive in case some columns bursting.
            cellsToSynapse = []
            for lastCell2 in self.lastActiveFCells:
                if self.FCells[ lastCell2 ].predictive:
                    cellsToSynapse.append( lastCell2 )
                elif not self.FCells[ lastCell2 ].predictive and lastCell2 % self.FCellsPerColumn == 0:
                    cellsToSynapse.append( self.SelectWinnerCell( int( lastCell2 / self.FCellsPerColumn ) ) )
            for thisCell in cellsToSynapse:
                self.FCells[ thisCell ].CreateSegment( winnerCells, self.initialPermanence, vector, self.initialPosVariance )

            # Create for winnerCells a new segment to lastActive cells.
            newVector = [ 0 ] * len( vector )
            for i in range( len( vector) ):
                newVector[ i ] = -vector[ i ]
            for winCell in winnerCells:
                if len( cellsToSynapse ) > 0:
                    self.FCells[ winCell ].CreateSegment( cellsToSynapse, self.initialPermanence, newVector, self.initialPosVariance )

#        # Add winner cell to object rep by creating synapses for all active OCells, and prime it.
#        for oCell in self.activeOCells:
#            self.OCells[ oCell ].NewSynapse( winnerCell, self.initialPermanence )
#        self.FCells[ winnerCell ].primed = True

#        toPrint = []
#        for cell in self.FCells:
#            toPrint.append( len( cell.segments ) )
#        print( toPrint )

    def ActivateFCells( self, columnSDR ):
    # Uses activated columns and cells in predictive state to put cells in active states.

        # Clean up old active cells and store old ones in lastActiveFCells.
        for aCell in self.activeFCells:
            self.FCells[ aCell ].active = False
        self.lastActiveFCells = self.activeFCells
        self.activeFCells = []

        activeCells  = []
        burstingCols = []

        for col in columnSDR.sparse:
            columnPredictive = False

            # Check if any cells in column are predictive. If yes then make them active.
            for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if self.FCells[ cell ].predictive:
                    columnPredictive = True
                    self.FCells[ cell ].active = True
                    activeCells.append( cell )

            # If none predictive then burst column, making all cells in column active.
            if not columnPredictive:
                burstingCols.append( col )
                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                    self.FCells[ cell ].active = True
                    activeCells.append( cell )

        print( "Bursting Pct: ", len( burstingCols ) / self.columnDimensions * 100, "%" )

        # Add the active cells to the list of active cells.
        self.activeFCells = sorted( activeCells )

        return burstingCols

    def PredictFCells( self, vector ):
    # Clear old predicted FCells and generate new predicted FCells.

        # Clean up old predictive cells.
        for pCell in self.predictiveFCells:
            self.FCells[ pCell ].predictive = False
        self.predictiveFCells = []

        predictFCells = []
        score         = [ 0 ] * self.columnDimensions * self.FCellsPerColumn

        for lastActive in self.activeFCells:
            for segment in self.FCells[ lastActive ].segments:
                if segment.Inside( vector ):
                    for pCell in segment.synapses:
                        score[ pCell ] += 1

        for indx, cell in enumerate( self.FCells ):
            if score[ indx ] >= self.activationThreshold:
                cell.predictive = True
                predictFCells.append( indx )

        self.predictiveFCells = sorted( predictFCells )

    def PrimeFCells( self ):
    # Go through active OCells and prime all FCells.

        # First reset old primed FCells.
        for prCell in self.primedFCells:
            self.FCells[ prCell ].primed = False
        self.primedFCells = []

        for oCell in self.activeOCells:
            for syn in self.OCells[ oCell ].synapses:
                self.FCells[ syn.FCell ].primed = True
                self.primedFCells.append( syn.FCell )

    def Compute( self, columnSDR, vectorX, vectorY ):
    # Compute the action of vector memory, and learn on the synapses.

        if len( self.activeFCells ) > 0:
            self.PredictFCells( [vectorX, vectorY ] )

        # Clear old active cells and get new ones active cells for this time step.
        burstingCols = self.ActivateFCells( columnSDR )

#        print( "Predicted: ", self.predictiveFCells )
#        print( "Active: ", self.activeFCells )

        # If there are enough columns bursting then we consider this a new feature, and not just noise.
        if len( burstingCols ) >= self.activationThreshold:
            self.NewFeature( burstingCols, [ vectorX, vectorY ] )
