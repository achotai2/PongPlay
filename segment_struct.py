from random import uniform, choice, sample, randrange, shuffle
from bisect import bisect_left
from collections import Counter
from useful_functions import BinarySearch, NoRepeatInsort, ModThisSynapse, IndexIfItsIn, FastIntersect, GenerateUnitySDR, NumStandardDeviations, CalculateDistanceScore, DelIfIn
import numpy
import math
#from time import time

class Segment:

    def __init__( self, vectorMemoryDict, incidentColumns, terminalColumns, incidentCells, terminalCells, vectorSDR, ID, confidenceScoreRank ):
    # Initialize the inidividual segment structure class.
    # Generate random permanence connections to all received incident cells, and to terminal cell.

        self.ID = ID

        self.vectorMemoryDict = vectorMemoryDict

        self.active              = True                        # True means it's predictive, and above threshold terminal cells fired.
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

        self.confidenceScore     = 0.0
        for perm in self.terminalPermanences:
            self.confidenceScore += perm
        self.confidenceScoreRank = confidenceScoreRank

        self.incidentActivation  = []           # A list of all columns that last overlapped with incidentSynapses.

        # Vector portion.
        self.vectorSynapses = vectorSDR.copy()

    def SetConfidenceRank( self, newRank ):
    # Set the confidenceRank.

        self.confidenceScoreRank = newRank

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
        self.confidenceScore += self.vectorMemoryDict[ "permanenceIncrement" ]

        if self.terminalPermanences[ terIndex ] >= 1.0:
            self.terminalPermanences[ terIndex ] = 1.0

    def DecreaseTerminalPermanence( self, terIndex ):
    # Decreases permanence of indexed terminal cell.

        self.terminalPermanences[ terIndex ] -= self.vectorMemoryDict[ "permanenceDecrement" ]
        self.confidenceScore -= self.vectorMemoryDict[ "permanenceDecrement" ]

        if self.terminalPermanences[ terIndex ] <= 0.0:
            # If it is below 0.0 then delete it.
            del self.terminalSynapses[ terIndex ]
            del self.terminalPermanences[ terIndex ]

        if len( self.terminalSynapses ) == 0:
            self.markedForDeletion = True

    def DecayTerminalPermanence( self, terIndex ):
    # Decreases permanence of indexed terminal cell.

        self.terminalPermanences[ terIndex ] -= self.vectorMemoryDict[ "permanenceDecay" ]
        self.confidenceScore -= self.vectorMemoryDict[ "permanenceDecay" ]

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

    def DecreaseIncidentPermanence( self, incIndex, FCells ):
    # Decreases permanence of indexed incident cell.

        self.incidentPermanences[ incIndex ] -= self.vectorMemoryDict[ "permanenceDecrement" ]

        if self.incidentPermanences[ incIndex ] <= 0.0:
            # If it is below 0.0 then delete it.

            thsad = self.incidentSynapses[incIndex]
            self.RemoveIncidentSynapse( incIndex, FCells )

    def RemoveIncidentSynapse( self, cellIndexToDelete, FCells ):
    # Delete the incident synapse sent.

        FCells[ self.incidentSynapses[ cellIndexToDelete ] ].DeleteIncidentSegmentReference( self.ID )

        del self.incidentSynapses[ cellIndexToDelete ]
        del self.incidentPermanences[ cellIndexToDelete ]
        del self.incidentColumns[ cellIndexToDelete ]

        if len( self.incidentSynapses ) == 0:
            self.markedForDeletion = True

    def NewIncidentSynapse( self, synapseToCreate, FCells ):
    # Create a new incident synapse.

        FCells[ synapseToCreate ].IncidentToThisSeg( self.ID )

        synLengthBefore = len( self.incidentSynapses )
        colLengthBefore = len( self.incidentColumns )

        indexToPut = NoRepeatInsort( self.incidentSynapses, synapseToCreate )
        self.incidentPermanences.insert( indexToPut, uniform( 0, self.vectorMemoryDict[ "initialPermanence" ] ) )
        NoRepeatInsort( self.incidentColumns, FCells[ synapseToCreate ].column )

        if len( self.incidentSynapses ) == synLengthBefore:
            print( "NewIncidentSynapse(): Tried to add an incident synapse, but already exists." )
            exit()
        if len( self.incidentColumns ) == colLengthBefore:
            print( "NewIncidentSynapse(): Tried to add an incident synapse, but synapse to this column already exists." )
            exit()

    def ModifyAllSynapses( self, FCells, lastActive ):
    # 1.) Decrease all terminal permanences to currently non-winner cells.
    # 2.) If terminal cell is winner increase permance for this terminal synapse;
    # 3.) Increase permanence strength to last-active incident cells that already have synapses;
    # 4.) Decrease synapse strength to inactive incident cells that already have synapses;
    # 5.) Build new synapses to active incident winner cells that don't have synapses;

        if self.active:
            # 1.)...
            for terIndex, terCell in enumerate( self.terminalSynapses ):
                if not FCells[ terCell ].winner:
                    self.DecreaseTerminalPermanence( terIndex )
            # 2.)...
                else:
                    self.IncreaseTerminalPermanence( terIndex )

            synapseToAdd = lastActive.copy()                          # Used for 5 below.

            for incIndex, incCell in enumerate( self.incidentSynapses ):
                # 3.)...
                if FCells[ incCell ].lastActive:
                    self.IncreaseIncidentPermanence( incIndex )
                    del synapseToAdd[ IndexIfItsIn( synapseToAdd, incCell ) ]
                # 4.)...
                elif not FCells[ incCell ].lastActive:
                    self.DecreaseIncidentPermanence( incIndex, FCells )

            # 5.)...
            if len( synapseToAdd ) > 0:
                # Check to make sure this segment doesn't already have a synapse to this column.
                realSynapsesToAdd = []

                for synAdd in synapseToAdd:
                    if FCells[ synAdd ].lastWinner and not BinarySearch( self.incidentColumns, FCells[ synAdd ].column ):
                        realSynapsesToAdd.append( synAdd )

                # Choose which synapses to actually add, since we can only add n-number per time step.
                reallyRealSynapsesToAdd = sample( realSynapsesToAdd, min( len( realSynapsesToAdd ), self.vectorMemoryDict[ "maxSynapsesToAddPer" ] ) )
                for toAdd in reallyRealSynapsesToAdd:
                    self.NewIncidentSynapse( toAdd, FCells )

                # If the number of synapses is above maxSynapsesPerSegment then delete random synapses.
                while self.vectorMemoryDict[ "maxSynapsesPerSegment" ] - len( self.incidentSynapses ) < 0:
                    toDelIndex = randrange( len( self.incidentSynapses ) )
                    self.RemoveIncidentSynapse( toDelIndex, FCells )

        else:
            if self.ReturnTerminalActivation() <= self.vectorMemoryDict[ "initialPermanence" ]:
                for terIndex, terCell in enumerate( self.terminalSynapses ):
                    self.DecayTerminalPermanence( terIndex )

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
        self.availableSegments  = []                                  # Stores the available segment numbers for new segments.
        for i in range( vectorMemoryDict[ "maxTotalSegments" ] ):
            self.segments.append( None )
            self.availableSegments.append( i )

        self.confidenceScoreRanks = []                                # Stores the confidence score rank for all segments.

        self.activeSegments     = []                                  # Stores currently active segments indices.

        self.numNonDeletedSegments = 0

    def HowManyActiveSegs( self ):
    # Return the number of active segments.

        return len( self.activeSegments )

    def HowManySegs( self ):
    # Return the number of segments.

        return self.numNonDeletedSegments

    def DeleteSegmentsAndSynapse( self, FCells ):
    # Goes through all segments and checks if they are marked for deletion. Deletes these segments from self.segments,
    # and removes all references to them in self.incidentSegments.

        for index, seg in enumerate( self.segments ):
            if seg != None and seg.markedForDeletion:
                # Delete any references to segment in FCells.
                for fCell in FCells:
                    fCell.DeleteIncidentSegmentReference( seg.ID )

                # Delete any references to this segment if they exist in activeSegments.
                DelIfIn( self.activeSegments, seg.ID )

                # Delete the entry from confidenceScoreRank
                del self.confidenceScoreRanks[ seg.confidenceScoreRank ]

                # Add entry index available again.
                NoRepeatInsort( self.availableSegments, seg.ID )

                # Delete the segment by setting the entry to None state.
                self.segments[ index ] = None

    def MarkSegmentForDeletion( self, segIndex ):
    # Mark the segment for deletion.

# CAN ACTUALLY JUST DELETE IT NOW I THINK, NO NEED TO MARK IT AND DELETE IT LATER, NO?
        if self.segments[ segIndex ] != None and self.segments[ segIndex ].MarkForDeletion():
            self.numNonDeletedSegments -= 1
        if self.segments[ segIndex ] == None:
            print( "MarkSegmentForDeletion(): Segment marked for deletion already doesn't exist." )
            exit()

    def CreateSegment( self, FCells, incidentColumns, terminalColumn, incidentCells, terminalCell, vectorSDR ):
    # Creates a new segment.

        # Get index and remove it from list.
        indexOfNew = self.availableSegments.pop( 0 )

        # Assemble and generate the new segment, and add it to list.
        terminalCells = []
        terminalCells.append( terminalCell )
        terminalColumns = []
        terminalColumns.append( terminalColumn )
        newSegment = Segment( self.vectorMemoryDict, incidentColumns, terminalColumns, incidentCells, terminalCells, vectorSDR, indexOfNew, None )
        if self.segments[ indexOfNew ] == None:
            self.segments[ indexOfNew ] = newSegment
        else:
            print( "CreateSegment(): Tried to create new segment but this entry was already full." )
            exit()

        # Add segment to active segments list.
        NoRepeatInsort( self.activeSegments, indexOfNew )

        # Update the segment confidenceScore ranking.
        self.InsertNewConfidenceRank( indexOfNew )

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
                    if segment != None and not segment.markedForDeletion and segment.ReturnTimeSinceActive() > longestInactiveTime:
                        longestInactiveIndex = segIndex
                        longestInactiveTime  = segment.ReturnTimeSinceActive()

                self.MarkSegmentForDeletion( longestInactiveIndex )

    def UpdateSegmentActivity( self, FCells ):
    # Make every segment that was active inactive, and refreshes its synapse activation.
    # Also add a time step to each segment, and see if it dies as a result. Delete any segments that die.
    # Also refresh FCells terminal activation.

        self.activeSegments = []

        # Increase time for all segments and deactivate them and alter their state.
        for index, segment in enumerate( self.segments ):
            if segment != None:
                segment.RefreshSegment()

        for fCell in FCells:
            fCell.RefreshTerminalActivation()

        # If number of segments is above total segment limit then choose the ones to delete.
        self.FindAndDeleteLongestInactive()

        self.DeleteSegmentsAndSynapse( FCells )

    def InsertNewConfidenceRank( self, segmentID ):
    # Given the segments confidence score input the newly created segment and update its confidence ranking.

        # Find the rank spot.
        newRank = 0
        while newRank < len( self.confidenceScoreRanks ) and self.segments[ self.confidenceScoreRanks[ newRank ] ].confidenceScore < self.segments[ segmentID ].confidenceScore:
            newRank += 1

        # Set the segments new rank in segment.
        self.segments[ segmentID ].SetConfidenceRank( newRank )

        # Shift all the higher ranks up by one.
        rankO = newRank
        while rankO < len( self.confidenceScoreRanks ):
            self.segments[ self.confidenceScoreRanks[ rankO ] ].SetConfidenceRank( self.segments[ self.confidenceScoreRanks[ rankO ] ].confidenceScoreRank + 1 )
            rankO += 1

        # Insert the segment into our list.
        if newRank < len( self.confidenceScoreRanks ):
            self.confidenceScoreRanks.insert( newRank, segmentID )
        else:
            self.confidenceScoreRanks.append( segmentID )

    def UpdateConfidenceRanks( self, segmentID ):
    # Update the ranking list of segments.
    # Lowest indices are lowest confidenceScore. Highest indices are highest confidenceScore.

        print( sorted( self.confidenceScoreRanks ) )
        for seg in self.segments:
            if seg != None and self.confidenceScoreRanks[ seg.confidenceScoreRank ] != seg.ID:
                print( seg.ID )
                print( self.confidenceScoreRanks[ seg.confidenceScoreRank ] )
                exit()

        # Loop until ranking is in correct spot.
        while True:
            if self.segments[ segmentID ].confidenceScoreRank > 0:
                earlierRankedSeg = self.confidenceScoreRanks[ self.segments[ segmentID ].confidenceScoreRank - 1 ]
            else:
                earlierRankedSeg = None
            if self.segments[ segmentID ].confidenceScoreRank < len( self.confidenceScoreRanks ) - 1:
                laterRankedSeg   = self.confidenceScoreRanks[ self.segments[ segmentID ].confidenceScoreRank + 1 ]
            else:
                laterRankedSeg   = None

            # Check if should move it back one.
            if earlierRankedSeg != None and self.segments[ segmentID ].confidenceScore < self.segments[ earlierRankedSeg ].confidenceScore:
                # Update list entries.
                del self.confidenceScoreRanks[ self.segments[ segmentID ].confidenceScoreRank ]
                self.confidenceScoreRanks.insert( self.segments[ segmentID ].confidenceScoreRank - 1, segmentID )

                # Update segment ranking reference.
                self.segments[ segmentID ].SetConfidenceRank( self.segments[ segmentID ].confidenceScoreRank - 1 )
                self.segments[ earlierRankedSeg ].SetConfidenceRank( self.segments[ segmentID ].confidenceScoreRank + 1 )

            # Check if should move it forward one.
            elif laterRankedSeg != None and self.segments[ segmentID ].confidenceScore > self.segments[ laterRankedSeg ].confidenceScore:
                # Update list entries.
                del self.confidenceScoreRanks[ self.segments[ segmentID ].confidenceScoreRank ]
                self.confidenceScoreRanks.insert( self.segments[ segmentID ].confidenceScoreRank + 1, segmentID )

                # Update segment ranking reference.
                self.segments[ segmentID ].SetConfidenceRank( self.segments[ segmentID ].confidenceScoreRank + 1 )
                self.segments[ laterRankedSeg ].SetConfidenceRank( self.segments[ segmentID ].confidenceScoreRank - 1 )

            # If it is in right spot then we leave it.
            else:
                break

    def SegmentLearning( self, FCells, lastActiveFCells ):
    # Perform learning on all active and inactive segments.
    # Refresh all segments then perform learning on them.
    # Delete segments that need deleting.

        # For all active segments, use the active incident cells and winner cells to modify synapses.
        for seg in self.segments:
            if seg != None:
                seg.ModifyAllSynapses( FCells, lastActiveFCells )
                self.UpdateConfidenceRanks( seg.ID )

#        self.CheckIfSegsIdentical( FCells )

#        for index, seg in enumerate(self.segments):
#            print( str(index) + ": " + str(seg.terminalPermanences))

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
        for segment in self.segments:
            if segment != None and segment.CheckActivation( vectorSDR ):                               # Checks incident overlap above threshold and if vector is inside.
                NoRepeatInsort( self.activeSegments, segment.ID )

                # Add to the terminal stimulation of the terminal cell.
                terminalCells, terminalPermanences = segment.GetTerminalSynapses()
                for index in range( len( terminalCells ) ):
                    # Terminal cell becomes predictive.
                    NoRepeatInsort( predictedCells, terminalCells[ index ] )
                    # Add strimulation to the cell.
                    FCells[ terminalCells[ index ] ].AddTerminalStimulation( terminalPermanences[ index ] )

        return predictedCells

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

        for segment in self.segments:
            if segment != None and segment.active:
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
                    segmentGroupings.append( [ segment.ID ] )
                else:
                    segmentGroupings[ chosenIndex ].append( segment.ID )

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
