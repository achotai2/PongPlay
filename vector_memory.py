from random import sample, randrange
from cell_and_synapse import FCell, Segment, BinarySearch, IndexIfItsIn, NoRepeatInsort, RepeatInsort, CheckInside
from operator import add
import numpy as np
from time import time

class VectorMemory:

    def __init__( self, columnDimensions, cellsPerColumn, numObjectCells, FActivationThresholdMin, FActivationThresholdMax, initialPermanence, lowerThreshold,
        permanenceIncrement, permanenceDecrement, permanenceDecay, segmentDecay, initialPosVariance, ObjectRepActivaton, OActivationThreshold,
        maxSegmentsPerCell, maxSynapsesPerSegment, maxSynapsesToAddPer, equalityThreshold, pctAllowedOCellConns ):

        self.columnDimensions        = columnDimensions         # Dimensions of the column space.
        self.FCellsPerColumn         = cellsPerColumn           # Number of cells per column.
        self.numObjectCells          = numObjectCells           # Number of cells in the Object level.
        self.FActivationThresholdMin = FActivationThresholdMin  # Threshold of active connected incident synapses...
                                                                # needed to activate segment.
        self.FActivationThresholdMax = FActivationThresholdMax  # Threshold of active connected incident synapses...
        self.initialPermanence       = initialPermanence        # Initial permanence of a new synapse.
        self.lowerThreshold          = lowerThreshold           # The lowest permanence for synapse to be active.
        self.permanenceIncrement     = permanenceIncrement      # Amount by which permanences of synapses are incremented during learning.
        self.permanenceDecrement     = permanenceDecrement      # Amount by which permanences of synapses are decremented during learning.
        self.permanenceDecay         = permanenceDecay          # Amount to decay permances each time step if < 1.0.
        self.segmentDecay            = segmentDecay             # If a segment hasn't been active in this many time steps then delete it.
        self.initialPosVariance      = initialPosVariance       # Amount of range vector positions are valid in.
        self.ObjectRepActivaton      = ObjectRepActivaton       # NNumber of active OCells in object layer at one time.
        self.OActivationThreshold    = OActivationThreshold     # Threshold of active connected OToFSynapses...
                                                                # needed to activate OCell.
        self.maxSegmentsPerCell      = maxSegmentsPerCell       # The maximum number of segments per cell.
        self.maxSynapsesPerSegment   = maxSynapsesPerSegment     # Maximum number of active bundles allowed on a segment.
        self.maxSynapsesToAddPer     = maxSynapsesToAddPer       # Maximum number of synapse bundles that can be added per time step.
        self.equalityThreshold       = equalityThreshold        # The number of equal synapses for two segments to be considered identical.
        self.pctAllowedOCellConns    = pctAllowedOCellConns     # Percent of OCells an FCell can build connections to.

        # --------Create all the cells in the network.---------

        self.columnSDR    = []
        self.burstingCols = []

        # Create cells in feature layer.
        self.FCells            = []
        for i in range( columnDimensions * cellsPerColumn ):
            self.FCells.append( FCell( initialPermanence, numObjectCells, pctAllowedOCellConns ) )

        self.lastActiveColumns = []

        # Create empty array for storing segments.
        self.segments        = []
        self.lastActiveSegs  = []
        self.activeSegs      = []
        self.deletedSegments = []

        # Create cells in object layer.
        self.activeOCells     = []
        self.lastActiveOCells = []

        self.workingMemory = []

    def ReturnSegmentData( self, timeStep ):
    # Checks on every segment and returns its state in a list.

        segment_data = []

        for index, segment in enumerate( self.segments ):
            this_seg_data = []
            this_seg_data.append( "------------------------------------------" )
            this_seg_data.append( "Time Step: " + str( timeStep ) + "\n" )

            returnedSegmentData = segment.GetSegmentData()
            for item in returnedSegmentData:
                this_seg_data.append( item )

            segment_data.append( this_seg_data )

        return segment_data

    def ReturnData( self ):
    # Return the active FCells as a list.

        activeFCells = []
        for fCell in self.FCells:
            activeFCells.append( fCell.active )

        return activeFCells, self.activeSegs, len( self.segments ) - len( self.deletedSegments )

    def ReturnActiveSegments( self ):
    # Return active segments as a list.

        return self.activeSegs

    def BuildLogData( self, log_data ):
    # Adds important information to log_data for entry into log.

        log_data.append( "Active Columns: " + str( len( self.columnSDR ) ) + ", " + str( self.columnSDR ) )

        activeFCells = []
        for index, fCell in enumerate( self.FCells ):
            if fCell.active:
                activeFCells.append( index )
        log_data.append( "Active F-Cells: " + str( len( activeFCells ) ) + ", " + str( activeFCells ) )

        log_data.append( "Bursting Column Pct: " + str( len( self.burstingCols ) / self.columnDimensions * 100 ) + "%, " + str( self.burstingCols ) )
#        log_data.append( "Active O-Cells: " + str( self.activeOCells ) )
        log_data.append( "Active Segments: " + str( len( self.activeSegs ) ) + ", ActiveSegs: " + str( self.activeSegs ) )
        log_data.append( "LastActive Segs: " + str( len( self.lastActiveSegs ) ) + ", LastActiveSegs: " + str( self.lastActiveSegs ) )

        predictedFCells = []
        for index, fCell in enumerate( self.FCells ):
            if fCell.predicted:
                predictedFCells.append( index )
        log_data.append( "Predicted Cells: " + str( len( predictedFCells ) ) + ", " + str( predictedFCells ) )

        nonDelSegments = []
        for index, seg in enumerate( self.segments ):
            if not seg.deleted:
                nonDelSegments.append( index )
        log_data.append( "Non-Deleted Segments: " + str( len( nonDelSegments) ) + ", " + str( nonDelSegments ) )

        log_data.append( "Deleted Segments: " + str( len( self.deletedSegments ) ) + ", " + str( self.deletedSegments ) )
        log_data.append( "Working Memory: " + str( self.workingMemory ) )

    def ActivateFCells( self ):
    # Uses activated columns and cells in predicted state to put cells in active states.

        # Clean up old active and lastActive FCells.
        for fCell in self.FCells:
            if fCell.lastActive:
                fCell.lastActive = False
                fCell.lastWinner = False

            if fCell.active:
                fCell.active     = False
                fCell.lastActive = True
                if fCell.winner:
                    fCell.winner     = False
                    fCell.lastWinner = True

        self.burstingCols = []

        for col in self.columnSDR:
            columnpredicted = False

            # Check if any cells in column are predicted. If yes then make them active.
            for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if self.FCells[ cell ].predicted:
                    columnpredicted = True
                    self.FCells[ cell ].active = True
                    self.FCells[ cell ].winner = True

            # If none predicted then burst column, making all cells in column active.
            # Choose a winner cell at random.
            if not columnpredicted:
                self.burstingCols.append( col )

#                minWinnersNum = -1
#                minWinnersIdx = 0
                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                    self.FCells[ cell ].active = True

#                    if minWinnersNum == -1 or self.FCells[ cell ].numWinners < minWinnersNum:
#                        minWinnersNum = self.FCells[ cell ].numWinners
#                        minWinnersIdx = cell

                self.FCells[ randrange( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ) ].winner = True

    def ActivateOCells( self ):
    # Use the active FCells to activate the OCells.

        # Refresh old active OCells.
        self.lastActiveOCells = self.activeOCells.copy()
        self.activeOCells = []

        oCellActivationLevel = [ 0 ] * self.numObjectCells
        # Use active FCells to decide on new active OCells.
        for index, fCell in enumerate( self.FCells ):
            if fCell.active:
                for index, oCell in enumerate( fCell.OCellConnections ):
                    if fCell.OCellPermanences[ index ] > self.lowerThreshold:
                        oCellActivationLevel[ oCell ] += 1

#        potentialActiveOcells = []
#        for index, oCell in enumerate( oCellActivationLevel ):
#            if oCell > self.OActivationThreshold:
#                potentialActiveOcells.append( ( oCell, index ) )
#        potentialActiveOcells.sort( reverse = True )

#        for a in range( self.ObjectRepActivaton ):
#            NoRepeatInsort( self.activeOCells, potentialActiveOcells[ a ][ 1 ] )

        for index, oCell in enumerate( oCellActivationLevel ):
            if oCell > self.OActivationThreshold:
                NoRepeatInsort( self.activeOCells, index )

    def IncrementSegTime( self, segsToDeleteList ):
    # Increment every OSegments timeSinceActive.

        for segIndx in range( len( self.segments ) ):
            if not self.segments[ segIndx ].deleted:
                self.segments[ segIndx ].timeSinceActive += 1

                if self.segments[ segIndx ].timeSinceActive > self.segmentDecay:
                    NoRepeatInsort( segsToDeleteList, segIndx )

    def DeleteSegments( self, segsToDeleteList ):
    # Deletes all segments in segsToDeleteList.

        for index in segsToDeleteList:
            self.segments[ index ].DeleteSegment()

            # Delete segments from activeSegs and lastActiveSegs if they are in there.
            actSegIndex = IndexIfItsIn( self.activeSegs, index )
            if actSegIndex != None:
                del self.activeSegs[ actSegIndex ]

            lActSegIndex = IndexIfItsIn( self.lastActiveSegs, index )
            if lActSegIndex != None:
                del self.lastActiveSegs[ lActSegIndex ]

            NoRepeatInsort( self.deletedSegments, index )

    def CreateSegment( self, vector ):
    # Create a new Segment, terminal on these active columns, incident on last active columns.

# THE PROBLEM WITH BELOW IS IF MORE THAN ONE CELL IN COLUMN IS ACTIVE (BECAUSE WE HYPOTHESIZE MULTIPLE FEATURES)
# THEN IT WILL CHOOSE THE ONE WITH THE LEAST TIMES WINNER TO BUILD SYNAPSE TO. BUT THIS WILL CROSS OVER FEATURE
# LINES AND MIX FEATURES. NOT WHAT WE WANT TO DO...

        terminalCellWinners = []
        for tCol in self.columnSDR:
            for cCell in range( tCol * self.FCellsPerColumn, ( tCol * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if self.FCells[ cCell ].winner:
                    terminalCellWinners.append( cCell )

        incidentCellWinners = []
        for iCol in self.lastActiveColumns:
            checkCells = []
            for cCell in range( iCol * self.FCellsPerColumn, ( iCol * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if self.FCells[ cCell ].lastWinner:
                    incidentCellWinners.append( cCell )

        newSegment = Segment( terminalCellWinners, incidentCellWinners, self.initialPermanence, self.columnDimensions, self.FCellsPerColumn, vector, self.initialPosVariance, self.FActivationThresholdMin )
        self.segments.append( newSegment )
        self.activeSegs.append( len( self.segments ) - 1 )

    def SegmentActivation( self, lastVector ):
    # Check if there are active segments terminating on above threshold # of active cells.
    # If none exist then the present result wasn't predicted; create a new segment and activate it.
    # De-activate all previously active segments.

        validActiveSeg   = []

        if len( self.segments ) > 0 and len( self.activeSegs ) > 0:
            for activeSegment in self.activeSegs:

                activeFCellsBool = []
                for fCell in self.FCells:
                    activeFCellsBool.append( fCell.active )

                if self.segments[ activeSegment ].CheckIfPredicting( activeFCellsBool, self.FCellsPerColumn, lastVector, self.lowerThreshold ):
                    NoRepeatInsort( validActiveSeg, activeSegment )
                    self.segments[ activeSegment ].timeSinceActive = 0

                else:
                    self.segments[ activeSegment ].active = False

        self.activeSegs = validActiveSeg

        if len( self.activeSegs ) == 0:
            self.CreateSegment( lastVector )

    def CheckIfSegsIdentical( self, segsToDelete ):
    # Compares all active segments to see if they have identical vectors and active synapse bundles.
    # If any do then delete one of them.

        if len( self.activeSegs ) > 1:
            for index1 in range( len( self.activeSegs ) - 1 ):
                for index2 in range( index1 + 1, len( self.activeSegs ) ):
                    actSeg1 = self.activeSegs[ index1 ]
                    actSeg2 = self.activeSegs[ index2 ]

                    if actSeg1 != actSeg2:
                        if self.segments[ actSeg1 ].Equality( self.segments[ actSeg2 ], self.equalityThreshold, self.lowerThreshold ):
                            NoRepeatInsort( segsToDelete, actSeg2 )
                    else:
                        print( "Error in CheckIfSegsIdentical()")
                        exit()

    def SegmentLearning( self, lastVector ):
    # Perform learning on segments, and create new ones if neccessary.

        segsToDelete = []
        primedSegments = []

        # Add time to all segments, and delete segments that haven't been active in a while.
        self.IncrementSegTime( segsToDelete )

        # Segment activation and create segments if none are active.
        self.SegmentActivation( lastVector )

        # If there is more than one segment active check if they are idential, if so delete one.
        self.CheckIfSegsIdentical( segsToDelete )

        # For every active and lastActive segment...
        if len( self.activeSegs ) > 0 and len( self.lastActiveSegs ) > 0:
            for activeSegIndex in self.activeSegs:

                # Gather the lastActive FCells in a bool list.
                lastActiveFCells = []
                for lCell in self.FCells:
                    lastActiveFCells.append( lCell.lastActive )

                # If active segments have positive synapses to non-active columns then decay them.
                # If active segments do not have terminal or incident synapses to active columns create them.
                self.segments[ activeSegIndex ].DecayAndCreate( lastActiveFCells, self.permanenceIncrement, self.permanenceDecrement, self.initialPermanence, self.maxSynapsesToAddPer, self.maxSynapsesPerSegment, self.FCellsPerColumn )

                # Adjust the segments activation threshold depending on number of winners selected.
                self.segments[ activeSegIndex ].AdjustThreshold( self.FActivationThresholdMin, self.FActivationThresholdMax )

        # Delete segments that had all their incident or terminal synapses removed.
        self.DeleteSegments( segsToDelete )

    def OCellLearning( self ):
    # Using the currently active and last active FCells, build stronger synapses to the active OCells,
    # and weaken to the losers.

        # Add all the active and last active OCells.
        allActiveOCells = []
        for actOCell in self.activeOCells:
            NoRepeatInsort( allActiveOCells, actOCell )
        for lActOCell in self.lastActiveOCells:
            NoRepeatInsort( allActiveOCells, lActOCell )

        # For all active and lastactive FCells strengthen to the active and lastActive OCells, and weaken to others.
        for actOCell in self.activeOCells:
            self.FCells[ actOCell ].OCellConnect( allActiveOCells, self.permanenceIncrement, self.permanenceDecrement )
        for lActOCell in self.lastActiveOCells:
            self.FCells[ lActOCell ].OCellConnect( allActiveOCells, self.permanenceIncrement, self.permanenceDecrement )

    def PredictFCells( self, vector ):
    # Clear old predicted FCells and generate new predicted FCells.

        # Clean up old predicted cells and segments.
        for fCell in self.FCells:
            if fCell.predicted:
                fCell.predicted = False

        self.activeSegs       = []
        self.lastActiveSegs   = []

        # Check every activeFCell's segments, and activate or deactivate them; make FCell predicted or not.
        for index in range( len( self.segments ) ):

            if not self.segments[ index ].deleted:

                # Make previously active segments lastActive (used in segment learning).
                if self.segments[ index ].active == True:
                    self.segments[ index ].lastActive = True
                    self.lastActiveSegs.append( index )
                else:
                    self.segments[ index ].lastActive = False

                activeFCellsBool = []
                for fCell in self.FCells:
                    activeFCellsBool.append( fCell.active )

                if self.segments[ index ].CheckIfPredicted( activeFCellsBool, self.FCellsPerColumn, vector, self.lowerThreshold ):

                    self.segments[ index ].active = True
                    self.activeSegs.append( index )

                    terminalFCellList = self.segments[ index ].ReturnTerminalFCells( self.FCellsPerColumn, self.lowerThreshold )

                    for cell in terminalFCellList:
                        self.FCells[ cell ].predicted = True

                else:
                    self.segments[ index ].active = False

        # If no segments are predicting cells then look into working memory if this vector is predicted there.
        wokePredicted = False
        if len( self.activeSegs ) == 0 and len( self.workingMemory ) > 0:
            for item in self.workingMemory:
                if CheckInside( [ 0, 0 ], list( map( add, item[1], vector ) ), self.initialPosVariance ):
                    for cell in item[ 0 ]:
                        wokePredicted = True
                        self.FCells[ cell ].predicted = True

    def UpdateWorkingMemory( self, vector ):
    # Use the vector to update all vectors stored in workingMemory items, and add timeStep. If any
    # item is above threshold time steps then remove it.
    # Add the new active Fcells to working memory.

        if len( self.workingMemory ) > 0:

            workingMemoryToDelete = []

            for index in range( len( self.workingMemory ) ):
                # Update vector.
                self.workingMemory[ index ][ 1 ][ 0 ] += vector[ 0 ]
                self.workingMemory[ index ][ 1 ][ 1 ] += vector[ 1 ]

# WILL HAVE TO WORK ON WORKING MEMORY TO MAKE IT MORE PRECISE.
                if self.workingMemory[ index ][ 1 ][ 0 ] == 0 and self.workingMemory[ index ][ 1 ][ 1 ] == 0:
                    workingMemoryToDelete.insert( 0, index )

                # Update time step.
                self.workingMemory[ index ][ 2 ] += 1

                if self.workingMemory[ index ][ 2 ] > self.segmentDecay:
                    if workingMemoryToDelete[ 0 ] != index:
                        workingMemoryToDelete.insert( 0, index )

            # Delete items whose timeStep is above threshold.
            if len( workingMemoryToDelete ) > 0:
                for toDel in workingMemoryToDelete:
                    del self.workingMemory[ toDel ]

        # Add active winner cells to working memory, with zero vector (since we're there), and no time steps,
        # if it doesn't exist.
        winnerCells = []
        for index, cell in enumerate( self.FCells ):
            if cell.winner:
                winnerCells.append( index )
        self.workingMemory.append( [ winnerCells, [ 0, 0 ], 0 ] )

    def Compute( self, columnSDR, lastVector, newVector ):
    # Compute the action of vector memory, and learn on the synapses.

        self.columnSDR = columnSDR.sparse.tolist()

        # Safety check for column dimensions.
        if columnSDR.size != self.columnDimensions:
            print( "VM input column dimensions but be same as input SDR dimensions." )
            exit()

        # Clear old active cells and get new ones active cells for this time step.
        self.ActivateFCells()

        # Update working memory vectors.
        self.UpdateWorkingMemory( lastVector )

        # Use the active FCells to tell us what OCells to activate.
        self.ActivateOCells()

        # Perform learning on OSegments.
        if len( self.lastActiveColumns ) > 0:
            self.SegmentLearning( lastVector )
#            self.OCellLearning()

        # Use FSegments to predict next set of inputs, given newVector.
        self.PredictFCells( newVector )

        self.lastActiveColumns = self.columnSDR
