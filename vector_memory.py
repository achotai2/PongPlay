from random import sample, randrange
from cell_and_synapse import FCell, OCell, Segment, BinarySearch, IndexIfItsIn, NoRepeatInsort, RepeatInsort
import numpy as np
from time import time

class VectorMemory:

    def __init__( self, columnDimensions, cellsPerColumn, numObjectCells, FActivationThreshold, initialPermanence,
        permanenceIncrement, permanenceDecrement, segmentDecay, initialPosVariance, OCellActivation, OActivationThreshold,
        maxNewFToFSynapses, maxSegmentsPerCell, maxNewOToFSynapses, equalityThreshold ):

        self.columnDimensions     = columnDimensions         # Dimensions of the column space.
        self.FCellsPerColumn      = cellsPerColumn           # Number of cells per column.
        self.numObjectCells       = numObjectCells           # Number of cells in the Object level.
        self.FActivationThreshold = FActivationThreshold     # Threshold of active connected incident synapses...
                                                             # needed to activate segment.
        self.initialPermanence    = initialPermanence        # Initial permanence of a new synapse.
        self.permanenceIncrement  = permanenceIncrement      # Amount by which permanences of synapses are incremented during learning.
        self.permanenceDecrement  = permanenceDecrement      # Amount by which permanences of synapses are decremented during learning.
        self.segmentDecay         = segmentDecay             # If a segment hasn't been active in this many time steps then delete it.
        self.initialPosVariance   = initialPosVariance       # Amount of range vector positions are valid in.
        self.OCellActivation      = OCellActivation          # Number of active OCells for object represention.
        self.OActivationThreshold = OActivationThreshold     # Threshold of active connected OToFSynapses...
                                                             # needed to activate OSegment.
        self.maxNewFToFSynapses   = maxNewFToFSynapses       # The maximum number of FToFSynapses added to a segment during creation.
        self.maxSegmentsPerCell   = maxSegmentsPerCell       # The maximum number of segments per cell.
        self.maxNewOToFSynapses   = maxNewOToFSynapses       # The maximum number of FToFSynapses added to a segment during creation.
        self.equalityThreshold    = equalityThreshold        # The number of equal synapses for two segments to be considered identical.

        # --------Create all the cells in the network.---------
        # Create cells in feature layer.
        self.FCells = []
        for i in range( columnDimensions * cellsPerColumn ):
            self.FCells.append( FCell( i ) )
        self.activeFCells      = []
        self.lastActiveFCells  = []
        self.lastActiveColumns = []
        self.predictiveFCells  = []

        # Create empty array for storing segments.
        self.segments       = []
        self.lastActiveSegs = []
        self.activeSegs     = []

        # Create cells in object layer.
        self.OCells = []
        for o in range( numObjectCells ):
            self.OCells.append( OCell( o ) )

    def ActivateFCells( self, columnSDR ):
    # Uses activated columns and cells in predictive state to put cells in active states.

        # Clean up old active cells and store old ones in lastActiveFCells.
        for aCell in self.activeFCells:
            self.FCells[ aCell ].active = False
        self.lastActiveFCells = self.activeFCells
        self.activeFCells = []

        burstingCols = []

        for col in columnSDR.sparse:
            columnPredictive = False

            # Check if any cells in column are predictive. If yes then make them active.
            for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if self.FCells[ cell ].predictive:
                    columnPredictive = True
                    self.FCells[ cell ].active = True
                    self.activeFCells.append( cell )

            # If none predictive then burst column, making all cells in column active.
            if not columnPredictive:
                burstingCols.append( col )
                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                    self.FCells[ cell ].active = True
                    NoRepeatInsort( self.activeFCells, cell )

        print( "Bursting Pct: ", len( burstingCols ) / self.columnDimensions * 100, "%" )

    def IncrementSegTime( self, segsToDeleteList ):
    # Increment every OSegments timeSinceActive.

        for segIndx in range( len( self.segments ) ):
            self.segments[ segIndx ].timeSinceActive += 1

            if self.segments[ segIndx ].timeSinceActive > self.segmentDecay:
                NoRepeatInsort( segsToDeleteList, segIndx )

    def DeleteSegments( self, segsToDeleteList ):
    # Deletes all segments in segsToDeleteList.

        for index in sorted( segsToDeleteList, reverse = True ):
            del self.segments[ index ]

    def CreateSegment( self, columnSDR, vector ):
    # Create a new Segment, terminal on these active columns, incident on last active columns.

        newSegment = Segment( columnSDR.sparse, self.lastActiveColumns, self.initialPermanence, self.columnDimensions, self.FCellsPerColumn, vector, self.initialPosVariance )
        self.segments.append( newSegment )
        self.activeSegs.append( len( self.segments ) - 1 )

    def SegmentActivation( self, columnSDR, lastVector ):
    # Check if there are active segments terminating on above threshold # of active cells.
    # If none exist then the present result wasn't predicted; create a new segment and activate it.
    # De-activate all previously active segments.

        validActiveSeg   = []

        if len( self.segments ) > 0 and len( self.activeSegs ) > 0:
            for activeSegment in self.activeSegs:

                # If the segment is overpredicted or overpredicting then we want to fracture it into two segments;
                # We can do this in two steps by fracturing the overpredicting first.
                if self.segments[ activeSegment ].numActiveTerminal >= len( columnSDR.sparse ) * 2 :
                    newSegment = self.segments[ activeSegment ].NonActiveTerminalCopy( columnSDR.sparse, self.initialPermanence, self.columnDimensions, self.FCellsPerColumn, lastVector, self.initialPosVariance )
                    self.segments.append( newSegment )
                elif self.segments[ activeSegment ].numActiveIncident >= len( columnSDR.sparse ) * 2 :
                    newSegment = self.segments[ activeSegment ].NonActiveIncidentCopy( self.lastActiveColumns, self.initialPermanence, self.columnDimensions, self.FCellsPerColumn, lastVector, self.initialPosVariance )
                    self.segments.append( newSegment )

# IMPLEMENT THE CREATION OF THE SEGMENT, THEN HAVE IT SO THAT IF MULTIPLE SEGS ARE ACTIVATING A PARTICULAR CELL
# ONLY THE ONE WITH THE HIGHEST TOTAL OVERLAP GETS TO COUNT THAT CELL FOR ITS OWN. BASICALLY, WHEN CALCULATING
# OVERLAP MAKE A LIST OF ALL THE OVERLAPPING CELLS; SORT THE LIST BY HIGHEST TOTAL OVERLAP TO LOWEST. THEN GO
# THROUGH THE LIST, STARTING WITH THE HIGHEST TOTAL OVERLAP SEGMENT, AND REMOVE ANY MORE INSTANCES OF A PARTICULAR
# CELL FROM ANY LOWER OVERLAP SEGS. THEN ONLY ACTIVATE THE SEGMENTS WITH ABOVE THRESHOLD OVERLAP AFTER THIS IS DONE.

                overlappingCells = self.segments[ activeSegment ].CheckIfPredicting( self.activeFCells, self.FCellsPerColumn, lastVector )

                if len( overlappingCells ) >= self.FActivationThreshold:
                    NoRepeatInsort( validActiveSeg, activeSegment )
                    self.segments[ activeSegment ].timeSinceActive = 0

                else:
                    self.segments[ activeSegment ].active = False

        self.activeSegs = validActiveSeg

        if len( self.activeSegs ) == 0:
            self.CreateSegment( columnSDR, lastVector )

    def CheckIfSegsIdentical( self, segsToDelete ):
    # Compares all active segments to see if they have identical vectors and active synapse bundles.
    # If any do then delete one of them.

        if len( self.activeSegs ) > 1:
            for actSeg1 in self.activeSegs:
                for actSeg2 in self.activeSegs:
                    if actSeg1 != actSeg2:
                        if self.segments[ actSeg1 ].Equality( self.segments[ actSeg2 ], self.equalityThreshold ):
                            NoRepeatInsort( segsToDelete, actSeg2 )

    def SegmentLearning( self, columnSDR, lastVector ):
    # Perform learning on segments, and create new ones if neccessary.

        segsToDelete = []

        # Add time to all segments, and delete segments that haven't been active in a while.
        self.IncrementSegTime( segsToDelete )

        # Segment activation and create segments if none are active.
        self.SegmentActivation( columnSDR, lastVector )

        # If there is more than one segment active check if they are idential, if so delete one.
        self.CheckIfSegsIdentical( segsToDelete )

        if len( self.activeSegs ) > 1:
            for i in self.activeSegs:
                print( self.segments[i])

        # For every active and lastActive segment...
        if len( self.activeSegs ) > 0 and len( self.lastActiveSegs ) > 0:

            for activeSegIndex in self.activeSegs:

                # If it has a positive synapses to non-active columns then decay them.
                # If it doesn't have them to active columns then create them.
                self.segments[ activeSegIndex ].DecayAndCreate( self.lastActiveColumns, columnSDR.sparse.tolist(), self.permanenceDecrement, self.initialPermanence )

            # For all last active columns find a winner cell on the terminal synapses of the last active segments,
            # and the incident synapses of the presently active segments.
            for lastActiveCol in self.lastActiveColumns:

                winnerCell = ( 0, 0.0 )     # ( Cell position in column, Strongest permanence value )

                # Go through all last active segments and look for winnerCell terminal on them in this column.
                for lastActiveSegIndx in self.lastActiveSegs:
                    winnerCellCheck = self.segments[ lastActiveSegIndx ].FindTerminalWinner( lastActiveCol, self.initialPermanence )
                    if winnerCellCheck[ 1 ] > winnerCell[ 1 ]:
                        winnerCell = winnerCellCheck

                # Go through all presently active segments and look for winnerCell incident on them in this column.
                # If any of these segments don't have any incident synapses at this column then create random ones.
                for activeSegIndex in self.activeSegs:
                    winnerCellCheck = self.segments[ activeSegIndex ].FindIncidentWinner( lastActiveCol, self.initialPermanence )
                    if winnerCellCheck[ 1 ] > winnerCell[ 1 ]:
                        winnerCell = winnerCellCheck

                # Now that we've found winner support the winner, and decay the loser, on this columns synapses.
                for lastActiveSegIndx in self.lastActiveSegs:
                    self.segments[ lastActiveSegIndx ].SupportTerminalWinner( lastActiveCol, winnerCell[ 0 ], self.permanenceIncrement, self.permanenceDecrement )

                for activeSegIndex in self.activeSegs:
                    self.segments[ activeSegIndex ].SupportIncidentWinner( lastActiveCol, winnerCell[ 0 ], self.permanenceIncrement, self.permanenceDecrement )

# HAVEN'T MADE IT SO THOSE WITH REMOVED SYNAPSES ARE ADDED TO segsToDelete.
# COULD ACTUALLY GET RID OF timeSinceActive, AND MAKE EVERY SYNAPSE BELOW 1.0 DECAY EVERY TIME STEP
# THEN WHEN ONE SEGMENT REACHES ALL 0.0 WE DELETE IT.

        # Delete segments that had all their incident or terminal synapses removed.
        self.DeleteSegments( segsToDelete )

    def PredictFCells( self, vector ):
    # Clear old predicted FCells and generate new predicted FCells.

        # Clean up old predictive cells.
        for pCell in self.predictiveFCells:
            self.FCells[ pCell ].predictive = False

        self.predictiveFCells = []
        self.activeSegs       = []
        self.lastActiveSegs   = []

        # Check every activeFCell's segments, and activate or deactivate them; make FCell predictive or not.
        for index in range( len( self.segments ) ):

            # Make previously active segments lastActive (used in segment learning).
            if self.segments[ index ].active == True:
                self.segments[ index ].lastActive = True
                self.lastActiveSegs.append( index )
            else:
                self.segments[ index ].lastActive = False

            if self.segments[ index ].CheckIfPredicted( self.activeFCells, self.FActivationThreshold, self.FCellsPerColumn, vector ):

                    self.segments[ index ].active = True
                    self.activeSegs.append( index )

                    terminalFCellList = self.segments[ index ].ReturnTerminalFCells( self.FCellsPerColumn )

                    for cell in terminalFCellList:
                        self.FCells[ cell ].predictive = True
                        NoRepeatInsort( self.predictiveFCells, cell )

            else:
                self.segments[ index ].active = False

    def Compute( self, columnSDR, lastVector, newVector ):
    # Compute the action of vector memory, and learn on the synapses.

        # Clear old active cells and get new ones active cells for this time step.
        self.ActivateFCells( columnSDR )

        # Perform learning on OSegments.
        if len( self.lastActiveColumns ) > 0:
            self.SegmentLearning( columnSDR, lastVector )

        print( "# of Active Segs: ", len( self.activeSegs ), ", ActiveSegs: ", self.activeSegs )
        print( "# of LastActive Segs: ", len( self.lastActiveSegs ), ", LastActiveSegs: ", self.lastActiveSegs )

        # Use FSegments to predict next set of inputs, given newVector.
        self.PredictFCells( newVector )

        self.lastActiveColumns = columnSDR.sparse.tolist()

#        toPrint = []
#        for seg in self.segments:
#            print( seg )
#        print( "FToFSynapse lengths: ", toPrint )
        print( "Active Cells: ", self.activeFCells )
        print( "Predictive Cells: ", self.predictiveFCells )
        print( "Number Active Cells: ", len( self.activeFCells ) )
        print( "Number Predictive Cells: ", len( self.predictiveFCells ) )
        print( "Number of Segments: ", len( self.segments ) )
