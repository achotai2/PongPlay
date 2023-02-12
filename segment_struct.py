from random import uniform, choice, sample, randrange, shuffle
from bisect import bisect_left
from collections import Counter
from useful_functions import BinarySearch, NoRepeatInsort, ModThisSynapse, IndexIfItsIn, FastIntersect, GenerateUnitySDR, NumStandardDeviations, CalculateDistanceScore, RemoveAndDecreaseIndices
import numpy
import math
#from time import time

class Segment:

    def __init__( self, vectorMemoryDict, incidentColumns, terminalColumns, incidentCells, terminalCells, vectorSDR ):
    # Initialize the inidividual segment structure class.
    # Generate random permanence connections to all received incident cells, and to terminal cell.

        self.vectorMemoryDict = vectorMemoryDict

        self.active              = False                         # True means it's predictive, and above threshold terminal cells fired.
        self.timeSinceActive     = 0                            # Time steps since this segment was active last.
        self.activationThreshold = vectorMemoryDict[ "FActivationThresholdMin" ]   # Minimum overlap required to activate segment.
        self.markedForDeletion   = False

        # Lateral synapse portion.
        self.incidentColumns     = incidentColumns.copy()
        self.incidentSynapses    = []
        self.incidentPermanences = []
        for iCell in incidentCells:
            self.incidentSynapses.append( iCell )
            self.incidentPermanences.append( uniform( 0, vectorMemoryDict[ "initialPermanence" ] ) )
        self.terminalColumns     = terminalColumns.copy()
        self.terminalSynapses    = []
        self.terminalPermanences = []
        for tCell in terminalCells:
            self.terminalSynapses.append( tCell )
            self.terminalPermanences.append( uniform( 0, vectorMemoryDict[ "initialPermanence" ] ) )

        self.incidentActivation  = []           # A list of all columns that last overlapped with incidentSynapses.

        # Vector portion.
        self.vectorSynapses = vectorSDR.copy()

    def Inside( self, vectorSDR ):
    # Checks if given vector position is inside range of number of standard deviations allowed.

        # Calculate vector probability that we are inside.
        intersection = FastIntersect( vectorSDR, self.vectorSynapses )

        if len( intersection ) >= self.activationThreshold:
            return True
        else:
            return False

    def IncidentCellActive( self, incidentCell, incidentColumn ):
    # Takes the incident cell and adds it to incidentActivation, if above permanenceLowerThreshold.
    # We actually add the column as we only want a single column activation to stimulate segment, if multiple cells per column are allowed.

        index = IndexIfItsIn( self.incidentSynapses, incidentCell )
        if index != None:
            if self.incidentPermanences[ index ] >= self.vectorMemoryDict[ "permanenceLowerThreshold" ]:
                NoRepeatInsort( self.incidentActivation, incidentColumn )

    def IncreaseTerminalPermanence( self, terIndex ):
    # Increases the permanence of indexed terminal cell.

        self.terminalPermanences[ terIndex ] += self.vectorMemoryDict[ "permanenceIncrement" ]

        if self.terminalPermanences[ terIndex ] >= 1.0:
            self.terminalPermanences[ terIndex ] = 1.0

    def DecreaseTerminalPermanence( self, terIndex ):
    # Decreases permanence of indexed terminal cell.

        self.terminalPermanences[ terIndex ] -= self.vectorMemoryDict[ "permanenceDecrement" ]

        if self.terminalPermanences[ terIndex ] <= 0.0:
            # If it is below 0.0 then delete it.
            del self.terminalSynapses[ terIndex ]
            del self.terminalPermanences[ terIndex ]

        if len( self.terminalSynapses ) == 0:
            self.markedForDeletion = True

    def IncreaseIncidentPermanence( self, incIndex ):
    # Increase permanence to indexed incident cell.

        self.incidentPermanences[ incIndex ] += self.vectorMemoryDict[ "permanenceIncrement" ]

        if self.incidentPermanences[ incIndex ] >= 1.0:
            self.incidentPermanences[ incIndex ] = 1.0

    def DecreaseIncidentPermanence( self, incIndex ):
    # Decreases permanence of indexed incident cell.

        self.incidentPermanences[ incIndex ] -= self.vectorMemoryDict[ "permanenceDecrement" ]

        if self.incidentPermanences[ incIndex ] <= 0.0:
            # If it is below 0.0 then delete it.
            del self.incidentSynapses[ incIndex ]
            del self.incidentPermanences[ incIndex ]

        if len( self.incidentSynapses ) == 0:
            self.markedForDeletion = True

    def ModifyAllSynapses( self, FCells ):
    # 1.) Decrease all terminal permanences to currently non-winner cells.
    # 2.) If terminal cell is winner increase permance for this terminal synapse;
    # 3.) Increase permanence strength to last-active incident cells that already have synapses;
    # 4.) Decrease synapse strength to inactive incident cells that already have synapses;
    # 5.) Build new synapses to active incident winner cells that don't have synapses;

            # 1.)...
            for terIndex, terCell in enumerate( self.terminalSynapses ):
                if not FCells[ terCell ].winner:
                    self.DecreaseTerminalPermanence( terIndex )
            # 2.)...
                else:
                    self.IncreaseTerminalPermanence( terIndex )

#            synapseToAdd = lastActiveFCells.copy()                          # Used for 5 above.

            for incIndex, incCell in enumerate( self.incidentSynapses ):
                # 3.)...
                if FCells[ incCell ].lastActive:
                    self.IncreaseIncidentPermanence( incIndex )
                # 4.)...
                else:
                    self.DecreaseIncidentPermanence( incIndex )

    def CheckActivation( self, vectorSDR ):
    # Checks the incidentActivation against activationThreshold to see if segment becomes active and stimulated.

        if len( self.incidentActivation ) >= self.activationThreshold and self.Inside( vectorSDR ):
            self.active            = True
            self.timeSinceActive   = 0

            return True

        self.active = False
        return False

    def GetTerminalSynapses( self ):
    # Return the terminal synapses and their permanence strengths.

        return self.terminalSynapses.copy(), self.terminalPermanences.copy()

    def ReturnTerminalActivation( self ):
    # Returns the sum of terminal permanences.

        sum = 0.0
        for permance in self.terminalPermanences:
            sum += permance

        return sum

    def RefreshSegment( self ):
    # Updates or refreshes the state of the segment.

        self.active             = False
        self.incidentActivation = []

        self.timeSinceActive += 1
        if self.vectorMemoryDict[ "segmentDecay" ] != -1 and self.timeSinceActive > self.vectorMemoryDict[ "segmentDecay" ]:
            self.markedForDeletion = True

    def RemoveIncidentSynapse( self, cellToDelete ):
    # Delete the incident synapse sent.

        index = IndexIfItsIn( self.incidentSynapses, cellToDelete )
        if index != None:
            del self.incidentSynapses[ index ]
            del self.incidentPermanences[ index ]
            del self.incidentColumns[ index ]
        else:
            print( "RemoveIncidentSynapse(): Attempt to remove synapse from segment, but synapse doesn't exist." )
            exit()

    def NewIncidentSynapse( self, synapseToCreate ):
    # Create a new incident synapse.

        index = bisect_left( self.incidentColumns, synapseToCreate )

        if index == len( self.incidentColumns ):
            self.incidentColumns.append( columnToCreate )
            for incCell in range( synapseToCreate * self.vectorMemoryDict[ "cellsPerColumn" ], ( synapseToCreate * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
                self.incidentSynapses.append( synapseToCreate )
                self.incidentPermanences.append( uniform( 0, 1 ) )

        elif self.incidentSynapses[ index ] != synapseToCreate:
            self.incidentColumns.insert( index, columnToCreate )
            for incCell in range( synapseToCreate * self.vectorMemoryDict[ "cellsPerColumn" ], ( synapseToCreate * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
                self.incidentSynapses.insert( ( index * self.vectorMemoryDict[ "cellsPerColumn" ] ) + incCell, synapseToCreate )
                self.incidentPermanences.insert( index * self.vectorMemoryDict[ "cellsPerColumn" ] + incCell, uniform( 0, 1 ) )

        else:
            print( "NewIncidentSynapse(): Attempt to add synapse to segment, but synapse to column already exists." )
            exit()

    def AlreadySynapseToColumn( self, checkSynapse ):
    # Checks if this segment has a synapse to the same column as checkSynapse.

        checkColumn = int( checkSynapse / self.vectorMemoryDict[ "cellsPerColumn" ] )

        if BinarySearch( self.incidentColumns, checkColumn ):
            return True
        else:
            return False

    def Equality( self, other ):
    # Runs the following comparison of equality: segment1 == segment2, comparing their activation intersection, but not vector.
# COULD RETURN JUST THE LENGTH OF INTERSECTION AND LET THE EQUALITY FUNCTION CHECK IF THIS IS A PROBLEM OR NOT...

        if len( FastIntersect( self.terminalSynapses, other.terminalSynapses ) ) == len( self.terminalSynapses ):
            if len( FastIntersect( self.incidentSynapses, other.incidentSynapses ) ) > self.vectorMemoryDict[ "equalityThreshold" ]:
                if len( FastIntersect( self.vectorSynapses, other.vectorSynapses ) ) > self.vectorMemoryDict[ "equalityThreshold" ]:
                    return True

        return False

    def CellForColumn( self, column ):
    # Return the cell for the incident column.

        index = IndexIfItsIn( self.incidentColumns, column )
        if index != None:
            return self.incidentSynapses[ index ]
        else:
            return None

    def ReturnTimeSinceActive( self ):
    # Return the time since this segment was last active.

        return self.timeSinceActive

    def MarkForDeletion( self ):
    # Mark this segment for deletion.

        if self.markedForDeletion:
            return False

        self.markedForDeletion = True
        return True

#-------------------------------------------------------------------------------

class SegmentStructure:

    def __init__( self, vectorMemoryDict ):
    # Initialize the segment storage and handling class.

        self.vectorMemoryDict = vectorMemoryDict

        self.segments           = []                                  # Stores all segments structures.
        self.activeSegments     = []

        self.numNonDeletedSegments = 0

    def HowManyActiveSegs( self ):
    # Return the number of active segments.

        return len( self.activeSegments )

    def DeleteSegmentsAndSynapse( self, FCells ):
    # Goes through all segments and checks if they are marked for deletion. Deletes these segments from self.segments,
    # and removes all references to them in self.incidentSegments.

        segsToDelete = []

        for index, seg in enumerate( self.segments ):
            if seg.markedForDeletion:
                segsToDelete.append( index )

        if len( segsToDelete ) > 0:
            for segIndex in reversed( segsToDelete ):
                # Delete any references to segment, and lower the index of all greater segment reference indices by one.
                for fCell in FCells:
                    fCell.DeleteIncidentSegmentReference( segIndex )

                # Delete any references to this segment if they exist, and modify indices.
                RemoveAndDecreaseIndices( self.activeSegments, segIndex )

                # Delete the segment.
                del self.segments[ segIndex ]

    def CreateSegment( self, FCells, incidentColumns, terminalColumn, incidentCells, terminalCell, vectorSDR ):
    # Creates a new segment.

        # Assemble and generate the new segment, and add it to list.
        terminalCells = []
        terminalCells.append( terminalCell )
        terminalColumns = []
        terminalColumns.append( terminalColumn )
        newSegment = Segment( self.vectorMemoryDict, incidentColumns, terminalColumns, incidentCells, terminalCells, vectorSDR )
        self.segments.append( newSegment )

        # Add segment to active segments list.
        indexOfNew = len( self.segments ) - 1
        self.activeSegments.append( indexOfNew )

        # Add reference to segment in incident cells.
        for incCell in incidentCells:
            FCells[ incCell ].IncidentToThisSeg( indexOfNew )

        self.numNonDeletedSegments += 1

    def FindAndDeleteLongestInactive( self ):
    # If there are too many segments find the segments that has been inactive longest and mark them for deletion.

        if self.vectorMemoryDict[ "maxTotalSegments" ] != -1:
            while self.numNonDeletedSegments > self.vectorMemoryDict[ "maxTotalSegments" ]:
                longestInactiveIndex = 0
                longestInactiveTime  = 0

                for segIndex, segment in enumerate( self.segments ):
                    if not segment.markedForDeletion and segment.ReturnTimeSinceActive() > longestInactiveTime:
                        longestInactiveIndex = segIndex
                        longestInactiveTime  = segment.ReturnTimeSinceActive()

                self.MarkSegmentForDeletion( longestInactiveIndex )

    def MarkSegmentForDeletion( self, segIndex ):
    # Mark the segment for deletion.

        if self.segments[ segIndex ].MarkForDeletion():
            self.numNonDeletedSegments -= 1

    def UpdateSegmentActivity( self, FCells ):
    # Make every segment that was active inactive, and refreshes its synapse activation.
    # Also add a time step to each segment, and see if it dies as a result. Delete any segments that die.
    # Also refresh FCells terminal activation.

        self.activeSegments = []

        # Increase time for all segments and deactivate them and alter their state.
        for index, segment in enumerate( self.segments ):
            segment.RefreshSegment()

        for fCell in FCells:
            fCell.RefreshTerminalActivation()

        # If number of segments is above total segment limit then choose the ones to delete.
        self.FindAndDeleteLongestInactive()

        self.DeleteSegmentsAndSynapse( FCells )

    def SegmentLearning( self, FCells ):
    # Perform learning on all active and inactive segments.
    # Refresh all segments then perform learning on them.
    # Delete segments that need deleting.

        # For all active segments, use the active incident cells and winner cells to modify synapses.
        for activeSeg in self.activeSegments:
            self.segments[ activeSeg ].ModifyAllSynapses( FCells )

        self.CheckIfSegsIdentical( FCells )

    def StimulateSegments( self, FCells, activeCells, vectorSDR ):
    # Using the activeCells and next vector find all segments that activate from this.
    # Use these active segments to get the predicted cells for given column.

        predictedCells = []
        potentialSegs  = []             # The segments that are stimulated and at zero vector.

        # First stimulate and activate any segments from the incident cells.
        # Activate the synapses in segments using activeCells.
        for incCell in activeCells:
            incCol, incSegments = FCells[ incCell ].ReturnIncidentOn()
            for entry in incSegments:
                self.segments[ entry ].IncidentCellActive( incCell, incCol )

        # Check the overlap of all segments and see which ones are active, and add the terminalCell to stimulatedCells.
        for segIndex, segment in enumerate( self.segments ):
            if segment.CheckActivation( vectorSDR ):                               # Checks incident overlap above threshold and if vector is inside.
                NoRepeatInsort( self.activeSegments, segIndex )

                # Add to the terminal stimulation of the terminal cell.
                terminalCells, terminalPermanences = segment.GetTerminalSynapses()
                for index in range( len( terminalCells ) ):
                    # Terminal cell becomes predictive.
                    NoRepeatInsort( predictedCells, terminalCells[ index ] )
                    # Add strimulation to the cell.
                    FCells[ terminalCells[ index ] ].AddTerminalStimulation( terminalPermanences[ index ] )

        return predictedCells

    def ChangePermanence( self, FCells, incCell, segIndex, permanenceChange ):
    # Change the permanence of synapse incident on incCell, part of segIndex, by permanenceChange.

        if FCells[ incCell ].ConnectionToSegment( segIndex ):
            self.segments.ModifyIncidentSynapse( incCell, permanenceChange )
        else:
            print("ChangePermanence(): Referenced cell does not have incident connection to referenced segment.")
            exit()

    def ThereCanBeOnlyOne( self, FCells, activeCellsList ):
    # Choose the winner cell for column by choosing active one with highest terminal activation.

        # Find the predicted cell
        greatestActivation = 0
        greatestCell       = activeCellsList[ 0 ]

        if len( activeCellsList ) > 1:
            for cell in activeCellsList:
                if FCells[ cell ].GetTerminalActivation() > greatestActivation:
                    greatestActivation = FCells[ cell ].GetTerminalActivation()
                    greatestCell       = cell

        return greatestCell

    def CheckIfSegsIdentical( self, Cells ):
    # Compares all segments to see if they have identical vectors or active synapse bundles. If any do then merge them.
    # A.) Begin by checking which segments are active.
    # B.) Group these segments by forming a list of lists, where each entry is a grouping of segments.
    #   Segments are grouped by first checking their terminal synapse against one, and then checking if their overlap is above equalityThreshold.
    # C.) Then, if any entry has more than two segments in it, merge the two, and mark one of the segments for deletion.

        segmentGroupings = []

        for index, segment in enumerate( self.segments ):
            if segment.active:
                chosenIndex = None
                for entryIndex, entry in enumerate( segmentGroupings ):
                    thisEntryMatches = True
                    for entrySegment in entry:
                        if not segment.Equality( self.segments[ entrySegment ] ):
                            thisEntryMatches = False
                            break
                    if thisEntryMatches:
                        chosenIndex = entryIndex
                        break

                if chosenIndex == None:
                    segmentGroupings.append( [ index ] )
                else:
                    segmentGroupings[ chosenIndex ].append( index )

# THIS COULD PROBABLY BE MADE BETTER BY MERGING THEM, RATHER THAN JUST DELETING ONE.
        for group in segmentGroupings:
            if len( group ) > 1:
                winnerSegmentIdx = 0
                winnerActivation = 0
                for gIndex, gSeg in enumerate( group ):
                    gActivation = self.segments[ gSeg ].ReturnTerminalActivation()
                    if gActivation > winnerActivation:
                        winnerSegmentIdx = gIndex
                        winnerActivation = gActivation

                group.pop( winnerSegmentIdx )

                for segIndex in group:
                    print("DELETED IDENTIAL SEGMENTS--------------------")
                    self.MarkSegmentForDeletion( segIndex )
