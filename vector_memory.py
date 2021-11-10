from random import sample, randrange
from cell_and_synapse import FCell, OCell, OSegment, BinarySearch, NoRepeatInsort, RepeatInsort
import numpy as np
from time import time
from bisect import bisect_left

class VectorMemory:

    def __init__( self, columnDimensions, cellsPerColumn, numObjectCells, FActivationThreshold, initialPermanence,
        permanenceIncrement, permanenceDecrement, segmentDecay, initialPosVariance, OCellActivation, OActivationThreshold,
        maxNewFToFSynapses, maxSegmentsPerCell, maxNewOToFSynapses ):

        self.columnDimensions     = columnDimensions         # Dimensions of the column space.
        self.FCellsPerColumn      = cellsPerColumn           # Number of cells per column.
        self.numObjectCells       = numObjectCells           # Number of cells in the Object level.
        self.FActivationThreshold = FActivationThreshold     # Threshold of active connected FToFSynapses...
                                                             # needed to activate FSegment.
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

        # --------Create all the cells in the network.---------
        # Create cells in feature layer.
        self.FCells = []
        for i in range( columnDimensions * cellsPerColumn ):
            self.FCells.append( FCell( i ) )
        self.activeFCells     = []
        self.lastActiveFCells = []
        self.lastActiveColumn = []
        self.predictiveFCells = []

        # Create empty array for storing OSegments.
        self.OSegments        = []

        # Create cells in object layer.
        self.OCells = []
        for o in range( numObjectCells ):
            self.OCells.append( OCell( o ) )
            # Create a random set of active object cells to start.
        self.activeOCells = sorted( sample( range( numObjectCells ), OCellActivation ) )

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

    def CreateOSegment( self, col, vector ):
    # Create a new OSegment, terminal on this FCell, with lateral synapses attached to incidentFCells,
    # and upwards synapses attached to attachedOCells.
    # Returns a list of the incidentFCell synapses created.

        # New segment terminal on all cells in column.
        terminalFCellList = list( range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ) )

        # New segment incident on all cells in all lastActive columns.
# THIS WILL CHANGE TO ONLY CERTAIN LAST ACTIVE COLUMNS.
        incidentFCellList = []
        for lACol in self.lastActiveColumn:
            for lACell in range( lACol * self.FCellsPerColumn, ( lACol * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                incidentFCellList.append( lACell )

        newSegment = OSegment( self.activeOCells, terminalFCellList, col, incidentFCellList, self.initialPermanence, vector, self.initialPosVariance )
        insertIdx = RepeatInsort( self.OSegments, newSegment )

    def IncrementSegTime( self, segsToDeleteList ):
    # Increment every OSegments timeSinceActive.

        for segIndx in range( len( self.OSegments ) ):
            self.OSegments[ segIndx ].timeSinceActive += 1

            if self.OSegments[ segIndx ].timeSinceActive >= self.segmentDecay:
                segsToDeleteList.append( segIndx )

    def DeleteSegments( self, segsToDeleteList ):
    # Deletes all segments in segsToDeleteList.

        for index in sorted( segsToDeleteList, reverse = True ):
            del self.OSegments[ index ]

    def AdjustSynapses( self, toDelete, cellList, winnerCell, permanenceList, checkCol ):
    # Adjust the synapse permancenes in permanenceList.

        for idx in range( len( cellList ) ):
            if cellList[ idx ] == winnerCell:
                if permanenceList[ idx ] < 1.0:
                    permanenceList[ idx ] += self.permanenceIncrement
                else:
                    permanenceList[ idx ] = 1.0
            elif checkCol == -1 or int( cellList[ idx ] / self.FCellsPerColumn ) == checkCol:
                if permanenceList[ idx ] > 0.0:
                    permanenceList[ idx ] -= self.permanenceDecrement
                else:
                    permanenceList[ idx ] = 0.0
                    toDelete.append( idx )

        return permanenceList

    def DeleteSynapses( self, synapsesToDelete, segsToDelete, synapseList1, synapseList2, segIndex ):

        if len( synapsesToDelete ) > 0:
            for index in sorted( synapsesToDelete, reverse = True ):
                del synapseList1[ index ]
                del synapseList2[ index ]
            # If there are no synapses of this type left then delete the segment.
            if len( synapseList1 ) == 0:
                segsToDelete.append( segIndex )

    def OSegmentLearning( self, columnSDR, lastVector ):
    # Perform learning on OSegments, and create new ones if neccessary.

# I'M NOT PUNISHING SEGMENTS THAT PREDICT INCORRECTLY. THESE SYNAPSES NEED TO BE PUNISHED, IN ADDITION TO LEARNING
# TOWARDS WINNER CELLS. THE SELECTING OF WINNER CELLS IS ALREADY THE REWARD FOR CORRECT PREDICTIONS.

        if len( self.OSegments ) > 0:
            segsToDelete = []

# 1.) GO THROUGH LASTACTIVE COLUMNS AND FIND THE SEGMENTS THAT TERMINATE ON THEM, THEN FIND THE PRESENTLY ACTIVE COLUMNS
#       THAT HAVE SEGMENTS INCIDENT ON THOSE COLUMNS. CHOOSE THE WINNER CELL FROM THAT COLUMN AND SUPPORT ITS SYNAPSES.
            # Look for last active OSegments (meaning it predicted) terminal on all last active columns.
            for lCol in self.lastActiveColumn:

                # For all segments terminal on this column:
                lColIndex = bisect_left( self.OSegments, lCol )
                while lColIndex < len( self.OSegments ) and self.OSegments[ lColIndex ].terminalColumn == lCol:
                    if self.OSegments[ lColIndex ].primed and self.OSegments[ lColIndex ].lastActive:
                        # If we find one then perform learning on this segment:
                        # Look at the OToFSynapses for the terminal FCells on the last active segment,
                        # and compare them to the FToFSynapses incident on the current active segment.
                        # Choose the strongest synapse FCell as the winner, and support all its synapses.
                        # The rest of the cells have their synapses decayed.
                        highestSynapse = 0.0
                        chosenCell     = lCol * self.FCellsPerColumn
# WANT TO BE A BIT CAREFUL HERE. WE WANT TO MAKE SURE THE SEGMENTS WE ARE INCIDENT ARE HAVE A STRONG OVERLAP
# OF OCELLS, MEANING SAME OBJECT REP. THIS ALSO MEANS THAT THE IMPLEMENTATION IS GOING TO RESULT IN DIFFERENT
# SEGMENTS TERMINAL ON THIS COLUMN WITH DIFFERENT CHOSEN WINNER CELLS; THIS ISN'T A PROBLEM EXACTLY, WE JUST
# HAVE TO MAKE SURE THAT IF THESE DIFFERENT SEGMENTS ATTACH TO THE SAME OBJECT CELLS, AND ARE CONSISTENTLY BEING
# ACTIVATED TOGETHER, THE OCELLS NEED TO CHOOSE A WINNER SEGMENT AND DECAY THE OTHER REDUNDANT ONE.
# IT ACTUALLY MIGHT BE A PROBLEM BECAUSE DIFFERENT WINNER CELLS WILL BE CHOSEN, AND ADJUSTED ON THE INCIDENT SYNAPSES
# BUT THIS WILL HAPPEN FOR ALL THE TERMINAL SEGMENTS; THEY MIGHT NOT FIND AN EQUILIBRIUM.
# THIS WOULD ONLY HAPPEN IF THE SEGMENTS HAVE DIFFERENT INCIDENT CELLS (OTHERWISE THEY SHOULDN'T EXIST AS DIFFERENT
# SEGNENTS), IN WHICH CASE WE ARE LOOKING AT A POTENTIAL FRACTURING OF THE FEATURE REP, SO IT COLLAPSING ONTO DIFFERENT
# UNSTABLE REPS MAKES SENSE.
                        for lTCell in range( len( self.OSegments[ lColIndex ].terminalFCells ) ):
                            if self.OSegments[ lColIndex ].OToFPermanences[ lTCell ] > highestSynapse:
                                highestSynapse = self.OSegments[ lColIndex ].OToFPermanences[ lTCell ]
                                chosenCell     = self.OSegments[ lColIndex ].terminalFCells[ lTCell ]

                        toAdjustPresent = []
                        for pCol in columnSDR.sparse:

                            pColIndex = bisect_left( self.OSegments, pCol )

                            while pColIndex < len( self.OSegments ) and self.OSegments[ pColIndex ].terminalColumn == pCol:
                                if self.OSegments[ pColIndex ].primed and self.OSegments[ pColIndex ].active:

                                    toAdjustPresent.append( pColIndex )

                                    for pCell in range( len( self.OSegments[ pColIndex ].FToFSynapses ) ):
                                        if int( self.OSegments[ pColIndex ].FToFSynapses[ pCell ] / self.FCellsPerColumn ) == lCol:
                                            if self.OSegments[ pColIndex ].FToFPermanences[ pCell ] > highestSynapse:

                                                highestSynapse = self.OSegments[ pColIndex ].FToFPermanences[ pCell ]
                                                chosenCell     = self.OSegments[ pColIndex ].FToFSynapses[ pCell ]

                                    #=-=-=-0==-0=-0=-0=-0=0
                                    self.OSegments[ pColIndex ].timeSinceActive = 0

                                pColIndex += 1

                        # Adjust terminal synapses going to the winner cell, and the loser cells.
                        toDelete = []
                        self.OSegments[ lColIndex ].OToFPermanences = self.AdjustSynapses(
                            toDelete, self.OSegments[ lColIndex ].terminalFCells, chosenCell,
                            self.OSegments[ lColIndex ].OToFPermanences, -1
                            )
                        # Delete those synapses that have decayed to 0.0.
                        self.DeleteSynapses(
                            toDelete, segsToDelete, self.OSegments[ lColIndex ].terminalFCells,
                            self.OSegments[ lColIndex ].OToFPermanences, lColIndex
                            )

                        for idx in toAdjustPresent:
                            toDelete = []
                            # Adjust incident synapses going to the winner cell, and the loser cells.
                            self.OSegments[ idx ].FToFPermanences = self.AdjustSynapses(
                                toDelete, self.OSegments[ idx ].FToFSynapses, chosenCell,
                                self.OSegments[ idx ].FToFPermanences, lCol )
                            # Delete those synapses that have decayed to 0.0.
                            self.DeleteSynapses(
                                toDelete, segsToDelete, self.OSegments[ idx ].FToFSynapses,
                                self.OSegments[ idx ].FToFPermanences, idx
                                )

                    lColIndex += 1

# 2.) GO THROUGH AND FIND ACTIVE SEGMENTS TERMINATING ON THE CURRENTLY ACTIVE COLUMNS. IF NONE THEN THIS MEANS THE
#       PRESENT RESULT WASN"T PREDICTED SO CREATE A NEW SEGMENT, AS ALREADY DONE.
            for col in columnSDR.sparse:
                colIndex = bisect_left( self.OSegments, col )
                thisColHasActiveSeg = False
                while colIndex < len( self.OSegments ) and self.OSegments[ colIndex ].terminalColumn == col:
                    if self.OSegments[ colIndex ].primed and self.OSegments[ colIndex ].active:
                        thisColHasActiveSeg = True
                    colIndex += 1

                # If no cells in column have active OSegments then burst create segments to active OCells.
                if not thisColHasActiveSeg:
                    self.CreateOSegment( col, lastVector )

            # Add time to all segments, and delete segments that haven't been active in a while.
            self.IncrementSegTime( segsToDelete )
            # Delete segments that had all their FToFSynapses's or terminalFCells removed.
# PROBABLY SHOULD MAKE IT SO SEGMENTS WITH ALL SYNAPSES AT 1.0 DONT GET DESTROYED, THEY ARE IN LONG TERM MEMORY.
            self.DeleteSegments( segsToDelete )

        # If there are no OSegments yet created then burst create for all active columns.
        elif len( self.lastActiveFCells ) > 0:
                for col in columnSDR.sparse:
                    self.CreateOSegment( col, lastVector )

    def PredictFCells( self, vector ):
    # Clear old predicted FCells and generate new predicted FCells.

        # Clean up old predictive cells.
        for pCell in self.predictiveFCells:
            self.FCells[ pCell ].predictive = False
        self.predictiveFCells = []

        # Check every activeFCell's OSegments, and activate or deactivate them; make FCell predictive or not.
        for seg in self.OSegments:
            # Make previously active segments lastActive (used in segment learning).
            if seg.primed:
                if seg.active == True:
                    seg.lastActive = True
                else:
                    seg.lastActive = False

                if ( seg.Inside( vector )
                    and seg.FCellOverlap( self.activeFCells, self.FActivationThreshold ) ):
                        seg.active = True

                        for termFCell in seg.terminalFCells:
                            self.FCells[ termFCell ].predictive = True
                            NoRepeatInsort( self.predictiveFCells, termFCell )

#                        if len( seg.terminalFCells ) >= 3:
#                            print("Cell: ", seg )
                else:
                    seg.active = False

    def Compute( self, columnSDR, lastVector, newVector ):
    # Compute the action of vector memory, and learn on the synapses.

        # Clear old active cells and get new ones active cells for this time step.
        self.ActivateFCells( columnSDR )

        # Perform learning on OSegments.
        self.OSegmentLearning( columnSDR, lastVector )

        # Use FSegments to predict next set of inputs, given newVector.
        self.PredictFCells( newVector )

        self.lastActiveColumn = columnSDR.sparse.tolist()

#        toPrint = []
#        for seg in self.OSegments:
#            toPrint.append( ( len( seg.FToFPermanences ), seg.timeSinceActive ) )
#        print( "FToFSynapse lengths: ", toPrint )
        print( "Number Active Cells: ", len( self.activeFCells ) )
        print( "Active Cells: ", self.activeFCells )
        print( "Number Predictive Cells: ", len( self.predictiveFCells ) )

        print("Number of OSegments: ", len( self.OSegments ) )
