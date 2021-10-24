from random import randrange, sample
from bisect import bisect_left
import numpy as np

def BinarySearch( list, val ):
# Search a sorted list.

    i = bisect_left( list, val )
    if i != len( list ) and list[ i ] == val:
        return True
    else:
        return False

class Synapse:

    def __init__( self, FCell, OCell, initialPermanance, posDimensions, posX, posY, posRange ):
    # Create a new synapse from Object OCell to Feature FCell.

        self.OCell  = OCell
        self.FCell  = FCell
        self.permanance    = initialPermanance
        self.posDimensions = posDimensions      # Number of dimensions vector positions will be in.
        self.positions     = []                 # A list of tuples, one for each dimension, with a range for location.
        for i in range( self.posDimensions ):
            posI = ( posX - posRange / 2, posX + posRange / 2 )
            self.positions.append( posI )

    def Inside( self, posX, posY ):
    # Checks if given position is inside range.

        if posX >= self.positions[ 0 ][ 0 ] and posX <= self.positions[ 0 ][ 1 ]:
            if posY >= self.positions[ 1 ][ 0 ] and posY <= self.positions[ 1 ][ 1 ]:
                return True
            else:
                return False
        else:
            return False

class FCell:

    def __init__( self, ID ):
    # Create a new inactive feature level cell with no synapses.

        self.ID             = ID
        self.active         = False           # Means column burst, or cell was predictive and then column fired.
        self.predictive     = False           # Means synapses on connected segments above activationThreshold.
        self.numSegments    = 0               # Number of OCell segments that are attached to this FCell.

class OCell:

    def __init__( self, ID ):
    # Create a new inactive feature level cell with no synapses.

        self.ID             = ID
        self.active         = False
        self.segments       = []              # Contains lists of synapses. Each segment attaches to a unique feature.
        self.activeSegments = []              # A list of indexes to segments that predicted FCells that then fired.

    def __lt__( self, other ):
    # Use for < comparison of cells. Used in sort algorithm.

         return len( self.segments ) < len( other.segments )

    def CreateSegment( self, FCellList, initialPermanence, posX, posY, posRange ):
    # Create a new segment with synapses connecting this OCell to all FCells in list, centered around position given.
    # Make that segment active.

        newSegment = []
        for FCell in FCellList:
            newSynapse = Synapse( FCell, self.ID, initialPermanence, 2, posX, posY, posRange )
            newSegment.append( newSynapse )

        self.segments.append( newSegment )

        self.activeSegments.append( len( self.segments ) - 1 )

    def SegmentActivation( self, activationThreshold, activeFCells, posX, posY ):
    # Check for any segments with synapses connecting to currently active Fcells. Activate any segments with
    # synapses above threshold, that also agree with vector location range (which is stored in the synapses).

        if len( self.segments ) > 0:
            for indx, segment in enumerate( self.segments ):
                segScore = 0
                for synapse in segment:
                    if BinarySearch( activeFCells, synapse.FCell ) and synapse.Inside( posX, posY ):
                        segScore += 1

#                    print(activeFCells)
#                    print(synapse.FCell)
                    if (BinarySearch( activeFCells, synapse.FCell )):
                        print ("It's True!")

                if segScore >= activationThreshold:
                    self.activeSegments.append( indx )
                    print(indx)

    def SegmentLearning( self, activeFCells ):
    # 2.) Increase permanance of synapses in active segments that connect to active cells, and decrease permanance
    #   of those synapses in active segments that don't. Also decrease slightly the permanance of inactive segments.
    # 3.) If no segments are sufficient to become active (meaning object specific unexpected input), and no other
    #   active OCells were predicting these FCells (meaning it is globally unexpected), then create a new segment
    #   with connections between this OCell and all these FCells, and the current vector position, and make it active.

        if len( self.segments ) > 0:
            noneSufficient = False
            for seg in self.segments:
                print("yo")

        return False            # Returns False if there are no active segments, so create a new segment.

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
        self.activeFCells = []

        # Create cells in object layer.
        self.OCells = []
        for o in range( numObjectCells ):
            self.OCells.append( OCell( o ) )

            # Create a random set of active object cells to start.
        self.activeOCells = sorted( sample( range( numObjectCells ), OCellActivation ) )

        self.burstingCols = []

    def SelectWinnerCells( self, burstingColumns ):
    # Selects one winner cell from each active columns, choosing the one with the least segments attached.

        winnerCells = []

        for col in burstingColumns:
            segsPerCell = []

            for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                numSegments = 0
                for Ocell in self.OCells:
                    for segment in Ocell.segments:
                        for synapse in segment:
                            if synapse.FCell == cell:
                                numSegments += 1
                                break
                segsPerCell.append( [ cell, numSegments ] )
            segsPerCell.sort( key = lambda segsPerCell: segsPerCell[ 1 ] )
            winnerCells.append( segsPerCell[ 0 ][ 0 ] )

        return winnerCells

    def ActivateFCells( self, columnSDR ):
    # Uses activated columns and cells in predictive state to put cells in active states.

        # De-activate old active cells and bursting columns in feature level.
        self.burstingCols = []
        for cell in self.activeFCells:
            self.FCells[ cell ].active = False

        activeCells = []

        for col in columnSDR.sparse:
            columnPredictive = False

            # Check if any cells in column are predictive. If yes then make them active.
            for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if self.FCells[ cell ].predictive:
                    columnPredictive = True
                    self.FCells[ cell ].active = True
                    activeCells.append( cell )

            # If none predictive then burst column.
            if not columnPredictive:
                self.burstingCols.append( col )
                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                    self.FCells[ cell ].active = True
                    activeCells.append( cell )

        print( "Bursting Pct: ", len( self.burstingCols ) / self.columnDimensions * 100, "%" )

        # Add the active cells to the list of active cells.
        self.activeFCells = sorted( activeCells )

#    def PredictCells( self, oldActive, newActive ):
    # Generate new predicted cells.

#    def ClearPrediction( self ):
    # Clear and refresh the predicted cells.

    def SynapseLearning( self, columnSDR ):
    # Perform learning on synpases by:

        for FCol in columnSDR.sparse:
            # 1.) Check if column is bursting.
            if FCol in self.burstingCols:
                # a.) If it is then we are seeing something unexpected. Choose a winner Fcell with least segments
                #       from the column. Build a segment from this Fcell to all active Ocells and activate segment.
                #       Also
                if len( self.activeOCells ) > self.OCellActivation:
                    for i in range( self.FCellsPerColumn ):
                        print("1a")

                # b.) If there aren't enough Ocells active above threshold then we are just starting observation.
                else:
                    solidSegs = False
                    leastSeg    = 0
                    leastSegNum = 0
                    for FCell in range( FCol * self.FCellsPerColumn, ( FCol * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                        # i.) Check if any Fcells in column have segments to Ocells, meaning we've already learned this object.
                        if len( self.FCells[ FCell ].segments ) > 0:
                            #   If yes then activate these segments.
                            solidSegs = True
                            print( "1bi" )

                        elif len( self.FCells[ FCell ].segments ) <= leastSegNum:
                            leastSeg    = FCell
                            leastSegNum = len( self.FCells[ FCell ].segments )

                    if not solidSegs:
                    # ii.) If none this means we haven't learned this feature. Choose Fcell with least segments as winner.
                    # Build a new segment from this cell to a set of random Ocells with least segments, and build segments
                    #  from Ocells back to Fcell. We're learning a new object represention.
                        OCellsLeastSegs = self.OCells.copy()
                        OCellsLeastSegs.sort()
                        while len( OCellsLeastSegs ) > self.OCellActivation:
                            OCellsLeastSegs.pop( -1 )
                        self.FCells[ leastSeg ].CreateSegment( OCellsLeastSegs )

        # 2.) If column is not bursting:
        #   a.) For all active Fcells in column, these were predicted by the object level. Activate their segments
        #       connecting back to currently active Ocells.

    def Compute( self, columnSDR ):
    # Generate active cells, clear last predicted cells, generate next predictive cells.

        # Clear old active cells and get new ones active cells for this time step.
        self.ActivateFCells( columnSDR )

        # For all active OCells:
        for actOCell in self.activeOCells:
            # Activate their segments.
            self.OCells[ actOCell ].SegmentActivation( self.activationThreshold, self.activeFCells, 0, 0 )
            # Perform learning on these active segments.
            if not self.OCells[ actOCell ].SegmentLearning( self.activeFCells ):
                # Or, create a new segment if none are active. Select winner cells from bursting columns.
                winnerCells = self.SelectWinnerCells( self.burstingCols )
                self.OCells[ actOCell ].CreateSegment( winnerCells, self.initialPermanence, 0, 0, self.initialPosVariance )
                print ("created segment")
        # Perform learning on pre-existing synapses.
#        if len( self.activeFCellBuffer ) > 1:
#            self.SynapseLearning( columnSDR )

        # Generate new predictive cells.
#        self.PredictCells( oldActive, newActive )
