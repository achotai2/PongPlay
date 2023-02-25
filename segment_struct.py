from random import uniform, choice, sample, randrange, shuffle
from bisect import bisect_left
from collections import Counter
from useful_functions import BinarySearch, NoRepeatInsort, ModThisSynapse, IndexIfItsIn, FastIntersect, GenerateUnitySDR, NumStandardDeviations, CalculateDistanceScore, DelIfIn
import numpy
import math
#from time import time

class Segment:

    def __init__( self, vectorMemoryDict, incidentColumns, terminalColumns, incidentCells, terminalCells, vectorSDR, ID ):
    # Initialize the inidividual segment structure class.
    # Generate random permanence connections to all received incident cells, and to terminal cell.

        self.ID = ID

        self.vectorMemoryDict = vectorMemoryDict

        self.active              = True                        # True means it's predictive, and above threshold terminal cells fired.
        self.winner              = False                       # Only the segment terminal on cell which has highest confidenceScore becomes winner.
        self.lastWinner          = False

        self.markedForDeletion   = False

# NEED TO MAKE THE THRESHOLDS DIFFERENT FOR EACH INPUT, AND ADJUST THEM.
        self.activationThreshold = vectorMemoryDict[ "FActivationThresholdMin" ]   # Minimum overlap required to activate segment.

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

        terBefore = self.terminalPermanences[ terIndex ]
        self.terminalPermanences[ terIndex ] += self.vectorMemoryDict[ "permanenceIncrement" ]

        if self.terminalPermanences[ terIndex ] >= 1.0:
            self.terminalPermanences[ terIndex ] = 1.0

        self.confidenceScore += self.terminalPermanences[ terIndex ] - terBefore

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

        if self.winner:
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
# PROBABLY BETTER TO REMOVE THE ONE WITH THE LOWEST PERMANENCE, THIS SHOULDNT BE HARD TO DO EITHER AS THERE IS A FUNCTION TO DO THIS.
                while self.vectorMemoryDict[ "maxSynapsesPerSegment" ] - len( self.incidentSynapses ) < 0:
                    toDelIndex = randrange( len( self.incidentSynapses ) )
                    self.RemoveIncidentSynapse( toDelIndex, FCells )

        else:
            if self.confidenceScore <= self.vectorMemoryDict[ "confidenceConfident" ]:
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

    def SetAsWinner( self ):
    # Set as winner.

        self.winner = True

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

        self.lastWinner         = self.winner
        self.winner             = False

        return self.markedForDeletion

    def CellForColumn( self, column ):
    # Return the cell for the incident column.

        index = IndexIfItsIn( self.incidentColumns, column )
        if index != None:
            return self.incidentSynapses[ index ]
        else:
            return None

    def MarkForDeletion( self ):
# CAN PROBABLY REMOVE.
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

        self.activeSegments     = []                                  # Stores currently active segments indices.
        self.winnerSegments     = []
        self.lastWinnerSegs     = []

    def HowManyActiveSegs( self ):
    # Return the number of active segments.

        return len( self.activeSegments )

    def HowManySegs( self ):
    # Return the number of segments.

        return self.vectorMemoryDict[ "maxTotalSegments" ] - len( self.availableSegments )

    def HowManyWinnerSegs( self ):
    # Return the number of winner segments.

        return len( self.winnerSegments )

    def DeleteSegmentsAndSynapse( self, FCells, segIndex ):
    # Goes through all segments and checks if they are marked for deletion. Deletes these segments from self.segments,
    # and removes all references to them in self.incidentSegments.

        if self.segments[ segIndex ] != None:
            # Delete any references to segment in FCells.
            for fCell in FCells:
                fCell.DeleteIncidentSegmentReference( segIndex )

            # Delete any references to this segment if they exist in activeSegments.
            DelIfIn( self.activeSegments, segIndex )

            # Add entry index available again.
            NoRepeatInsort( self.availableSegments, segIndex )

            # Delete the segment by setting the entry to None state.
            self.segments[ segIndex ] = None

    def CreateSegment( self, FCells, incidentColumns, terminalColumn, incidentCells, terminalCell, vectorSDR ):
    # Creates a new segment.

        if len( self.availableSegments ) == 0:
            print( "CreateSegment(): Want to create new segment, but none left available." )
            exit()

        # Get index and remove it from list.
        indexOfNew = self.availableSegments.pop( 0 )

        # Assemble and generate the new segment, and add it to list.
        terminalCells = []
        terminalCells.append( terminalCell )
        terminalColumns = []
        terminalColumns.append( terminalColumn )
        newSegment = Segment( self.vectorMemoryDict, incidentColumns, terminalColumns, incidentCells, terminalCells, vectorSDR, indexOfNew )
        if self.segments[ indexOfNew ] == None:
            self.segments[ indexOfNew ] = newSegment
        else:
            print( "CreateSegment(): Tried to create new segment but this entry was already full." )
            exit()

        # Add segment to active segments list.
        NoRepeatInsort( self.activeSegments, indexOfNew )

        # Add reference to segment in incident cells.
        for incCell in incidentCells:
            FCells[ incCell ].IncidentToThisSeg( indexOfNew )

    def UpdateSegmentActivity( self, FCells ):
    # Make every segment that was active inactive, and refreshes its synapse activation.
    # Also add a time step to each segment, and see if it dies as a result. Delete any segments that die.
    # Also refresh FCells terminal activation.

        self.activeSegments = []
        self.lastWinnerSegs = self.winnerSegments.copy()
        self.winnerSegments = []

        toDelete            = []

# MIGHT BE FASTER TO NOT HAVE TO CYCLE THROUGH EVERY SEQUENCE IF I CAN, AS THIS TAKES MORE TIME.
        # Increase time for all segments and deactivate them and alter their state.
        for index, segment in enumerate( self.segments ):
            if segment != None:
                if segment.RefreshSegment():
                    toDelete.append( index )

        for fCell in FCells:
            fCell.RefreshTerminalActivation()

        # Delete segments with no terminal or incident synapses.
        for delSeg in toDelete:
            self.DeleteSegmentsAndSynapse( FCells, delSeg )

    def SegmentLearning( self, FCells, lastActiveFCells ):
    # Perform learning on all active and inactive segments.
    # Refresh all segments then perform learning on them.
    # Delete segments that need deleting.

        # For all active segments, use the active incident cells and winner cells to modify synapses.
        for seg in self.segments:
            if seg != None:
                seg.ModifyAllSynapses( FCells, lastActiveFCells )

#        for index, seg in enumerate(self.segments):
#            print( str(index) + ": " + str(seg.terminalPermanences))

    def StimulateSegments( self, FCells, activeCells, vectorSDR ):
    # Using the activeCells and next vector find all segments that activate from this.
    # Use these active segments to get the predicted cells for given column.

        predictedCells = []             # All the cells terminal on segments which become active due to vectorSDR and active incident cells.
        segsForTCells  = []             # The segs which are terminal on the predictedCells.
        potentialSegs  = []             # The segments that are stimulated and at zero vector.

        # First stimulate and activate any segments from the incident cells.
        # Activate the synapses in segments using activeCells.
        for incCell in activeCells:
            incCol, incSegments = FCells[ incCell ].ReturnIncidentOn()
            for entry in incSegments:
                self.segments[ entry ].IncidentCellActive( incCell, incCol )

        # Check the overlap of all segments and see which ones are active, and add the terminalCell to stimulatedCells.
        for segment in self.segments:
            if segment != None and segment.CheckActivation( vectorSDR ):                   # Checks incident overlap above threshold and if vector is inside.
                NoRepeatInsort( self.activeSegments, segment.ID )

                # Add to the terminal stimulation of the terminal cell.
                terminalCells, terminalPermanences = segment.GetTerminalSynapses()

                for index in range( len( terminalCells ) ):
                    # Terminal cell becomes predictive.
                    insertIndex = NoRepeatInsort( predictedCells, terminalCells[ index ] )

                    # Add the segments for use in choosing the winner segment for each predicted cell.
                    if len( predictedCells ) == len( segsForTCells ):
                        NoRepeatInsort( segsForTCells[ insertIndex ], segment.ID )
                    else:
                        segsForTCells.insert( insertIndex, [ segment.ID ] )

                    # Add strimulation to the cell.
                    FCells[ terminalCells[ index ] ].AddTerminalStimulation( terminalPermanences[ index ] )

        # Choose the winner segment for each terminal cell.
        for entry in segsForTCells:
            highestConfidenceSeg   = entry[ 0 ]
            highestConfidenceValue = self.segments[ entry[ 0 ] ].confidenceScore

            for segIdx in entry:
                if self.segments[ segIdx ].confidenceScore > highestConfidenceValue:
                    highestConfidenceSeg   = segIdx
                    highestConfidenceValue = self.segments[ segIdx ].confidenceScore

            # Set this segment as the winner.
            self.segments[ highestConfidenceSeg ].SetAsWinner()
            NoRepeatInsort( self.winnerSegments, highestConfidenceSeg )

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
