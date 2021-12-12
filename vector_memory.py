from random import sample, randrange
from cell_and_synapse import FCell, Segment, BinarySearch, IndexIfItsIn, NoRepeatInsort, RepeatInsort
import numpy as np
from time import time

class VectorMemory:

    def __init__( self, columnDimensions, cellsPerColumn, numObjectCells, FActivationThresholdMin, FActivationThresholdMax, initialPermanence, lowerThreshold,
        permanenceIncrement, permanenceDecrement, permanenceDecay, segmentDecay, initialPosVariance, ObjectRepActivaton, OActivationThreshold,
        maxNewFToFSynapses, maxSegmentsPerCell, maxBundlesPerSegment, maxBundlesToAddPer, equalityThreshold, pctAllowedOCellConns ):

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
        self.maxNewFToFSynapses      = maxNewFToFSynapses       # The maximum number of FToFSynapses added to a segment during creation.
        self.maxSegmentsPerCell      = maxSegmentsPerCell       # The maximum number of segments per cell.
        self.maxBundlesPerSegment    = maxBundlesPerSegment     # Maximum number of active bundles allowed on a segment.
        self.maxBundlesToAddPer      = maxBundlesToAddPer       # Maximum number of synapse bundles that can be added per time step.
        self.equalityThreshold       = equalityThreshold        # The number of equal synapses for two segments to be considered identical.
        self.pctAllowedOCellConns    = pctAllowedOCellConns     # Percent of OCells an FCell can build connections to.

        # --------Create all the cells in the network.---------

        self.columnSDR    = []
        self.burstingCols = []

        # Create cells in feature layer.
        self.activeFCells      = []
        self.lastActiveFCells  = []
        self.FCells            = []
        for i in range( columnDimensions * cellsPerColumn ):
            self.FCells.append( FCell( initialPermanence, numObjectCells, pctAllowedOCellConns ) )

        self.lastActiveColumns = []
        self.predictiveFCells  = []

        # Create empty array for storing segments.
        self.segments        = []
        self.lastActiveSegs  = []
        self.activeSegs      = []
        self.deletedSegments = []

        # Create cells in object layer.
        self.activeOCells        = []
        self.lastActiveOCells    = []

#    def ReturnCellsAndSynapses( self, cell_report_data, segment_report_data ):
#    # Prepares details on the cell and synapse end state to print into the log file.
#
#        for index in range( len( self.FCells ) ):
#            cell_report_data.append( "----------------------------" )
#            cell_report_data.append( "FCell #: " + str( index ) )
#            cell_report_data.append( str( self.FCells[ index ] ) )
#
#        for index in range( len( self.segments ) ):
#            segment_report_data.append( "----------------------------" )
#            segment_report_data.append( "Segment #: " + str( index ) )
#            segment_report_data.append( str( self.segments[ index ] ) )

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

    def BuildLogData( self, log_data ):
    # Adds important information to log_data for entry into log.

        log_data.append( "Active Columns: " + str( len( self.columnSDR ) ) + ", " + str( self.columnSDR ) )
        log_data.append( "Active F-Cells: " + str( len( self.activeFCells ) ) + ", " + str( self.activeFCells ) )
        log_data.append( "Bursting Column Pct: " + str( len( self.burstingCols ) / self.columnDimensions * 100 ) + "%, " + str( self.burstingCols ) )
#        log_data.append( "Active O-Cells: " + str( self.activeOCells ) )
        log_data.append( "Active Segments: " + str( len( self.activeSegs ) ) + ", ActiveSegs: " + str( self.activeSegs ) )
        log_data.append( "LastActive Segs: " + str( len( self.lastActiveSegs ) ) + ", LastActiveSegs: " + str( self.lastActiveSegs ) )
        log_data.append( "Predictive Cells: " + str( len( self.predictiveFCells ) ) + ", " + str( self.predictiveFCells ) )
        nonDelSegments = []
        for index, seg in enumerate( self.segments ):
            if not seg.deleted:
                nonDelSegments.append( index )
        log_data.append( "Non-Deleted Segments: " + str( len( nonDelSegments) ) + ", " + str( nonDelSegments ) )
        log_data.append( "Deleted Segments: " + str( len( self.deletedSegments ) ) + ", " + str( self.deletedSegments ) )

    def ActivateFCells( self ):
    # Uses activated columns and cells in predictive state to put cells in active states.

        # Clean up old active cells and store old ones in lastActiveFCells.
        self.lastActiveFCells = self.activeFCells
        self.activeFCells = []

        self.burstingCols = []

        for col in self.columnSDR:
            columnPredictive = False

            # Check if any cells in column are predictive. If yes then make them active.
            for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if BinarySearch( self.predictiveFCells, cell ):
                    columnPredictive = True
                    self.activeFCells.append( cell )

            # If none predictive then burst column, making all cells in column active.
            if not columnPredictive:
                self.burstingCols.append( col )
                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                    NoRepeatInsort( self.activeFCells, cell )

    def ActivateOCells( self ):
    # Use the active FCells to activate the OCells.

        # Refresh old active OCells.
        self.lastActiveOCells = self.activeOCells.copy()
        self.activeOCells = []

        oCellActivationLevel = [ 0 ] * self.numObjectCells
        # Use active FCells to decide on new active OCells.
        for actFCell in self.activeFCells:
            for index, oCell in enumerate( self.FCells[ actFCell ].OCellConnections ):
                if self.FCells[ actFCell ].OCellPermanences[ index ] > self.lowerThreshold:
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

#        for index in sorted( segsToDeleteList, reverse = True ):
#            del self.segments[ index ]
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

        newSegment = Segment( self.columnSDR, self.lastActiveColumns, self.initialPermanence, self.columnDimensions, self.FCellsPerColumn, vector, self.initialPosVariance, self.FActivationThresholdMin )
        self.segments.append( newSegment )
        self.activeSegs.append( len( self.segments ) - 1 )

    def SegmentActivation( self, lastVector ):
    # Check if there are active segments terminating on above threshold # of active cells.
    # If none exist then the present result wasn't predicted; create a new segment and activate it.
    # De-activate all previously active segments.

        validActiveSeg   = []

        if len( self.segments ) > 0 and len( self.activeSegs ) > 0:
            for activeSegment in self.activeSegs:

                if self.segments[ activeSegment ].CheckIfPredicting( self.activeFCells, self.FCellsPerColumn, lastVector, self.lowerThreshold ):
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

            for lActiveSeg in self.lastActiveSegs:
                # For all lastActive segments support connection to currently active segments.
                self.segments[ lActiveSeg ].AddToPrime( self.activeSegs, self.initialPermanence, self.permanenceIncrement, self.permanenceDecay )

                # Prime all segments connected from the lastActive segments.
                newPrimed = self.segments[ lActiveSeg ].GetPrimed()
                for item in newPrimed:
                    NoRepeatInsort( primedSegments, item )

            print( "Primed Segments: " + str( primedSegments ) )
            print( "Last Active Segments: " + str( self.lastActiveSegs ) )
            print( "Active Segments: " + str( self.activeSegs ) )




            for activeSegIndex in self.activeSegs:
                # If active segments have positive synapses to non-active columns then decay them.
                # If active segments do not have terminal or incident synapses to active columns create them.
                self.segments[ activeSegIndex ].DecayAndCreateBundles( self.lastActiveColumns, self.columnSDR, self.permanenceDecay, self.initialPermanence, self.maxBundlesToAddPer )
                # Adjust the segments activation threshold depending on number of winners selected.
                self.segments[ activeSegIndex ].AdjustThreshold( self.FActivationThresholdMin, self.FActivationThresholdMax )

            # For all last active columns find a winner cell on the terminal synapses of the last active segments,
            # and the incident synapses of the presently active segments.
            for lastActiveCol in self.lastActiveColumns:

                winnerCell = ( 0, 0.0 )     # ( Cell position in column, Strongest permanence value )

                # Go through all last active segments and look for winnerCell terminal on them in this column.
                for lastActiveSegIndx in self.lastActiveSegs:
                    winnerCellCheck = self.segments[ lastActiveSegIndx ].FindTerminalWinner( lastActiveCol )
                    if winnerCellCheck[ 1 ] > winnerCell[ 1 ]:
                        winnerCell = winnerCellCheck

                # Go through all presently active segments and look for winnerCell incident on them in this column.
                # If any of these segments don't have any incident synapses at this column then create random ones.
                for activeSegIndex in self.activeSegs:
                    winnerCellCheck = self.segments[ activeSegIndex ].FindIncidentWinner( lastActiveCol )
                    if winnerCellCheck[ 1 ] > winnerCell[ 1 ]:
                        winnerCell = winnerCellCheck

                # Now that we've found winner support the winner, and decay the loser, on this columns synapses.
                for lastActiveSegIndx in self.lastActiveSegs:
                    self.segments[ lastActiveSegIndx ].SupportTerminalWinner( lastActiveCol, winnerCell[ 0 ], self.permanenceIncrement, self.permanenceDecrement )

                for activeSegIndex in self.activeSegs:
                    self.segments[ activeSegIndex ].SupportIncidentWinner( lastActiveCol, winnerCell[ 0 ], self.permanenceIncrement, self.permanenceDecrement )

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

        # Clean up old predictive cells and segments.
        self.predictiveFCells = []
        self.activeSegs       = []
        self.lastActiveSegs   = []

        # Check every activeFCell's segments, and activate or deactivate them; make FCell predictive or not.
        for index in range( len( self.segments ) ):

            if not self.segments[ index ].deleted:
                # Make previously active segments lastActive (used in segment learning).
                if self.segments[ index ].active == True:
                    self.segments[ index ].lastActive = True
                    self.lastActiveSegs.append( index )
                else:
                    self.segments[ index ].lastActive = False

                if self.segments[ index ].CheckIfPredicted( self.activeFCells, self.FCellsPerColumn, vector, self.lowerThreshold ):

                        self.segments[ index ].active = True
                        self.activeSegs.append( index )

                        terminalFCellList = self.segments[ index ].ReturnTerminalFCells( self.FCellsPerColumn, self.lowerThreshold )

                        for cell in terminalFCellList:
                            NoRepeatInsort( self.predictiveFCells, cell )

                else:
                    self.segments[ index ].active = False

    def Compute( self, columnSDR, lastVector, newVector ):
    # Compute the action of vector memory, and learn on the synapses.

        self.columnSDR = columnSDR.sparse.tolist()

        # Safety check for column dimensions.
        if columnSDR.size != self.columnDimensions:
            print( "VM input column dimensions but be same as input SDR dimensions." )
            exit()

        # Clear old active cells and get new ones active cells for this time step.
        self.ActivateFCells()

        # Use the active FCells to tell us what OCells to activate.
        self.ActivateOCells()

        # Perform learning on OSegments.
        if len( self.lastActiveColumns ) > 0:
            self.SegmentLearning( lastVector )
#            self.OCellLearning()

        # Use FSegments to predict next set of inputs, given newVector.
        self.PredictFCells( newVector )

        self.lastActiveColumns = self.columnSDR
