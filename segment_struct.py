from random import uniform, choice, sample, randrange, shuffle
from bisect import bisect_left
from collections import Counter
from useful_functions import SmoothStep, BinarySearch, NoRepeatInsort, ModThisSynapse, IndexIfItsIn, FastIntersect, GenerateUnitySDR, NumStandardDeviations, CalculateDistanceScore, DelIfIn
import numpy
import math
from time import time

class Segment:

    def __init__( self, vectorMemoryDict, incidentColumns, terminalColumns, incidentCells, terminalCells, vectorSDR, ID ):
    # Initialize the inidividual segment structure class.
    # Generate random permanence connections to all received incident cells, and to terminal cell.

        self.ID = ID

        self.vectorMemoryDict = vectorMemoryDict

        self.active              = True                        # True means it's predictive, and above threshold terminal cells fired.
        self.winner              = False                       # Only the segment terminal on cell which has highest confidenceScore becomes winner.

        self.markedForDeletion   = False

# NEED TO MAKE THE THRESHOLDS DIFFERENT FOR EACH INPUT, AND ADJUST THEM.
        self.activationThreshold = vectorMemoryDict[ "FActivationThresholdMin" ]   # Minimum overlap required to activate segment.
        self.incidentActivation  = []           # A list of all columns that last overlapped with incidentSynapses.

        # Lateral synapse portion.
        self.incidentColumns     = incidentColumns.copy()
        self.incidentSynapses    = incidentCells.copy()
        self.incidentPermanences = []
        for iCell in incidentCells:
            self.incidentPermanences.append( uniform( 0, vectorMemoryDict[ "initialPermanence" ] ) )
        self.terminalColumns     = terminalColumns.copy()
        self.terminalSynapses    = terminalCells.copy()
        self.terminalPermanences = []
        for tCell in terminalCells:
            self.terminalPermanences.append( uniform( 0, vectorMemoryDict[ "initialPermanence" ] ) )

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

    def IncreaseTerminalPermanence( self, FCells, terCell ):
    # Increases the permanence of indexed terminal cell.

        terIndex = IndexIfItsIn( self.terminalSynapses, terCell )

        self.terminalPermanences[ terIndex ] += self.vectorMemoryDict[ "permanenceIncrement" ]

        if self.terminalPermanences[ terIndex ] >= 1.0:
            self.terminalPermanences[ terIndex ] = 1.0

    def DecreaseTerminalPermanence( self, FCells, terCell ):
    # Decreases permanence of indexed terminal cell.

        terIndex = IndexIfItsIn( self.terminalSynapses, terCell )

        self.terminalPermanences[ terIndex ] -= self.vectorMemoryDict[ "permanenceDecrement" ]

        if self.terminalPermanences[ terIndex ] <= 0.0:
            # If it is below 0.0 then delete it.
            self.RemoveTerminalSynapse( FCells, terCell )

    def DecayTerminalPermanence( self, FCells, terCell ):
    # Decreases permanence of indexed terminal cell.

        terIndex = IndexIfItsIn( self.terminalSynapses, terCell )

# MAYBE JUST BETTER TO USE ModifyAllSynapses FUNCTION FOR ALL THESE, AND THEN WE CAN DECAY DIFFERENT AMOUNT FOR ACTIVE BUT NOT WINNER, AND NON ACTIVE SEGS.
        self.terminalPermanences[ terIndex ] -= self.vectorMemoryDict[ "permanenceDecay" ]

        if self.terminalPermanences[ terIndex ] <= 0.0:
            # If it is below 0.0 then delete it.
            self.RemoveTerminalSynapse( FCells, terCell )

    def IncreaseIncidentPermanence( self, FCells, incCell ):
    # Increase permanence to indexed incident cell.

        incIndex = IndexIfItsIn( self.incidentSynapses, incCell )

        self.incidentPermanences[ incIndex ] += self.vectorMemoryDict[ "permanenceIncrement" ]

        if self.incidentPermanences[ incIndex ] >= 1.0:
            self.incidentPermanences[ incIndex ] = 1.0

    def DecreaseIncidentPermanence( self, FCells, incCell ):
    # Decreases permanence of indexed incident cell.

        incIndex = IndexIfItsIn( self.incidentSynapses, incCell )

        self.incidentPermanences[ incIndex ] -= self.vectorMemoryDict[ "permanenceDecrement" ]

        if self.incidentPermanences[ incIndex ] <= 0.0:
            # If it is below 0.0 then delete it.
            self.RemoveIncidentSynapse( FCells, incCell )

    def RemoveTerminalSynapse( self, FCells, cellToDelete ):
    # Delete the terminal synapse sent.

        FCells[ cellToDelete ].DeleteTerminalSegmentReference( self.ID )

        cellIndexToDelete = IndexIfItsIn( self.terminalSynapses, cellToDelete )
        del self.terminalSynapses[ cellIndexToDelete ]
        del self.terminalColumns[ cellIndexToDelete ]
        del self.terminalPermanences[ cellIndexToDelete ]

        if len( self.terminalSynapses ) == 0:
            self.markedForDeletion = True

    def NewTerminalSynapse( self, FCells, synapseToCreate ):
    # Create a new incident synapse.

        FCells[ synapseToCreate ].TerminalToThisSeg( self.ID )

        synLengthBefore = len( self.terminalSynapses )
        colLengthBefore = len( self.terminalColumns )

        indexToPut = NoRepeatInsort( self.terminalSynapses, synapseToCreate )
        self.terminalPermanences.insert( indexToPut, uniform( 0, self.vectorMemoryDict[ "initialPermanence" ] ) )
        NoRepeatInsort( self.terminalColumns, FCells[ synapseToCreate ].column )

        if len( self.terminalSynapses ) == synLengthBefore:
            print( "NewIncidentSynapse(): Tried to add an incident synapse, but already exists." )
            exit()
        if len( self.terminalColumns ) == colLengthBefore:
            print( "NewIncidentSynapse(): Tried to add an incident synapse, but synapse to this column already exists." )
            exit()

    def RemoveIncidentSynapse( self, FCells, cellToDelete ):
    # Delete the incident synapse sent.

        FCells[ cellToDelete ].DeleteIncidentSegmentReference( self.ID )

        cellIndexToDelete = IndexIfItsIn( self.incidentSynapses, cellToDelete )
        del self.incidentSynapses[ cellIndexToDelete ]
        del self.incidentPermanences[ cellIndexToDelete ]
        del self.incidentColumns[ cellIndexToDelete ]

        if len( self.incidentSynapses ) <= self.vectorMemoryDict[ "FActivationThresholdMin" ]:
            self.markedForDeletion = True

    def NewIncidentSynapse( self, FCells, synapseToCreate ):
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

    def ModifyAllSynapses( self, FCells, active, lastActive ):
    # 1.) Decrease all terminal permanences to currently non-winner cells.
    # 2.) If terminal cell is winner increase permance for this terminal synapse;
    # 3.) Increase permanence strength to last-active incident cells that already have synapses;
    # 4.) Decrease synapse strength to inactive incident cells that already have synapses;
    # 5.) Build new synapses to active incident winner cells that don't have synapses;
    # 6.) Build new synapses to active terminal winner cells that don't have synapses;

        if self.winner:
            incSynapseToAdd = lastActive.copy()                          # Used for 5 below.
            terSynapseToAdd = active.copy()                          # Used for 6 below.

            # 1.)...
            for terCell in self.terminalSynapses:
                if not FCells[ terCell ].winner:
                    self.DecreaseTerminalPermanence( FCells, terCell )
            # 2.)...
                else:
                    self.IncreaseTerminalPermanence( FCells, terCell )

                if FCells[ terCell ].active:
                    del terSynapseToAdd[ IndexIfItsIn( terSynapseToAdd, terCell ) ]

            for incCell in self.incidentSynapses:
                # 3.)...
                if FCells[ incCell ].lastActive:
                    self.IncreaseIncidentPermanence( FCells, incCell )
                    del incSynapseToAdd[ IndexIfItsIn( incSynapseToAdd, incCell ) ]
                # 4.)...
                elif not FCells[ incCell ].lastActive:
                    self.DecreaseIncidentPermanence( FCells, incCell )

            # 5.)...
            if len( incSynapseToAdd ) > 0:
                # Check to make sure this segment doesn't already have a synapse to this column.
                realSynapsesToAdd = []

                for synAdd in incSynapseToAdd:
                    if FCells[ synAdd ].lastWinner and not BinarySearch( self.incidentColumns, FCells[ synAdd ].column ):
                        realSynapsesToAdd.append( synAdd )

                # Choose which synapses to actually add, since we can only add n-number per time step.
                reallyRealSynapsesToAdd = sample( realSynapsesToAdd, min( len( realSynapsesToAdd ), self.vectorMemoryDict[ "maxSynapsesToAddPer" ] ) )
                for toAdd in reallyRealSynapsesToAdd:
                    self.NewIncidentSynapse( FCells, toAdd )

                # If the number of synapses is above maxSynapsesPerSegment then delete random synapses.
# PROBABLY BETTER TO REMOVE THE ONE WITH THE LOWEST PERMANENCE, THIS SHOULDNT BE HARD TO DO EITHER AS THERE IS A FUNCTION TO DO THIS.
                while self.vectorMemoryDict[ "maxSynapsesPerSegment" ] - len( self.incidentSynapses ) < 0:
                    toDelIndex = randrange( len( self.incidentSynapses ) )
                    self.RemoveIncidentSynapse( FCells, self.incidentSynapses[ toDelIndex ] )

            # 6.)...
            if len( terSynapseToAdd ) > 0:
                # Check to make sure this segment doesn't already have a synapse to this column.
                realSynapsesToAdd = []

                for synAdd in terSynapseToAdd:
                    if FCells[ synAdd ].winner and not BinarySearch( self.terminalColumns, FCells[ synAdd ].column ):
                        realSynapsesToAdd.append( synAdd )

                # Choose which synapses to actually add, since we can only add n-number per time step.
                reallyRealSynapsesToAdd = sample( realSynapsesToAdd, min( len( realSynapsesToAdd ), self.vectorMemoryDict[ "maxSynapsesToAddPer" ] ) )
                for toAdd in reallyRealSynapsesToAdd:
                    self.NewTerminalSynapse( FCells, toAdd )

                # If the number of synapses is above maxSynapsesPerSegment then delete random synapses.
# PROBABLY BETTER TO REMOVE THE ONE WITH THE LOWEST PERMANENCE, THIS SHOULDNT BE HARD TO DO EITHER AS THERE IS A FUNCTION TO DO THIS.
                while self.vectorMemoryDict[ "maxSynapsesPerSegment" ] - len( self.incidentSynapses ) < 0:
                    toDelIndex = randrange( len( self.terminalSynapses ) )
                    self.RemoveTerminnalSynapse( FCells, self.incidentSynapses[ toDelIndex ] )

        # Decay terminal synapses if it is active but not winner.
        elif self.active:
            if not self.IsConfident():
                for terCell in self.terminalSynapses:
                    self.DecayTerminalPermanence( FCells, terCell )

        # Also Decay terminal synapses if it is not active but the terminal cell is active.
        else:
            if not self.IsConfident():
                for terCell in self.terminalSynapses:
                    self.DecayTerminalPermanence( FCells, terCell )

        # Update segments confidence score.
        self.UpdateConfidenceScore()

    def IsConfident( self ):
    # Checks if this segments confidence score is above threshold.

        if self.ReturnTerminalActivation() >= self.vectorMemoryDict[ "confidenceConfident" ]:
            return True
        else:
            return False

    def ReturnConfidenceScore( self ):
    # Returns the confidenceScore of this segment.

        return self.ReturnTerminalActivation()

    def UpdateConfidenceScore( self ):
    # Updates this segments confidence score and activation thresholds.

        # Get a score between 0.0 and 1.0.
        score = SmoothStep( self.ReturnTerminalActivation(), 0, 1, 2 )

        # Calculate the incident activation threshold given above score.
        self.activationThreshold = int( score * ( self.vectorMemoryDict[ "FActivationThresholdMax" ] - self.vectorMemoryDict[ "FActivationThresholdMin" ] ) + self.vectorMemoryDict[ "FActivationThresholdMin" ] )

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

    def GetIncidentSynapses( self ):
    # Return the terminal synapses and their permanence strengths.

        return self.incidentSynapses.copy(), self.incidentPermanences.copy()

    def ReturnTerminalActivation( self ):
    # Returns the sum of terminal permanences.

        sum = 0.0
        for permanence in self.terminalPermanences:
            sum += permanence

        return sum

    def RefreshSegment( self ):
    # Updates or refreshes the state of the segment.

        self.active             = False
        self.winner             = False
        self.incidentActivation = []

        return self.markedForDeletion

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
            incidentCells, incidentPermanences = self.segments[ segIndex ].GetIncidentSynapses()
            for iCell in incidentCells:
                FCells[ iCell ].DeleteIncidentSegmentReference( segIndex )
            terminalCells, terminalPermanences = self.segments[ segIndex ].GetTerminalSynapses()
            for tCell in terminalCells:
                FCells[ tCell ].DeleteTerminalSegmentReference( segIndex )

            # Delete any references to this segment if they exist in activeSegments.
            DelIfIn( self.activeSegments, segIndex )

            # Add entry index available again.
            NoRepeatInsort( self.availableSegments, segIndex )

            # Delete the segment by setting the entry to None state.
            self.segments[ segIndex ] = None

    def CreateSegment( self, FCells, incidentColumns, terminalColumns, incidentCells, terminalCells, vectorSDR ):
    # Creates a new segment.

        # If no segments available then randomly delete one below confidene threshold.
        while len( self.availableSegments ) == 0:
            checkIdx = randrange( self.vectorMemoryDict[ "maxTotalSegments" ] )
            if not self.segments[ checkIdx ].IsConfident():
                self.DeleteSegmentsAndSynapse( FCells, checkIdx )
#            print( "CreateSegment(): Want to create new segment, but none left available." )
#            exit()

        # Get index and remove it from list.
        indexOfNew = self.availableSegments.pop( 0 )

        # Assemble and generate the new segment, and add it to list.
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
            # Add segment but also check if incident cell has too many segments attached.
            if FCells[ incCell ].IncidentToThisSeg( indexOfNew ):
                # If if does then remove the segment with the lowest confidence.
                col, asIncident  = FCells[ incCell ].ReturnIncidentOn()
                lowestConfidence = 0.0
                lowestSeg        = asIncident[ 0 ]
                for seg in asIncident:
                    if self.segments[ seg ].ReturnConfidenceScore() < lowestConfidence:
                        lowestConfidence = self.segments[ seg ].ReturnConfidenceScore()
                        lowestSeg        = seg

                # Remove reference from both segment to FCell, and FCell to segment.
                self.segments[ lowestSeg ].RemoveIncidentSynapse( FCells, incCell )

        # Add reference to segment in terminal cells.
        for tCell in terminalCells:
            FCells[ tCell ].TerminalToThisSeg( indexOfNew )

    def UpdateSegmentActivity( self, FCells ):
    # Make every segment that was active inactive, and refreshes its synapse activation.
    # Also add a time step to each segment, and see if it dies as a result. Delete any segments that die.
    # Also refresh FCells terminal activation.

        toDelete = []

        # Refresh state of all active and winner segments.
        for actSeg in self.activeSegments:
            if self.segments[ actSeg ].RefreshSegment():
                toDelete.append( actSeg )

        for fCell in FCells:
            fCell.RefreshTerminalActivation()

        # Delete segments with no terminal or incident synapses.
        for delSeg in toDelete:
            self.DeleteSegmentsAndSynapse( FCells, delSeg )

        self.activeSegments = []
        self.winnerSegments = []

    def SegmentLearning( self, FCells, activeFCells, lastActiveFCells ):
    # Perform learning on all active and inactive segments.
    # Refresh all segments then perform learning on them.
    # Delete segments that need deleting.

        toModSegs = []

# MAYBE NOT NECCESSARY?
        # For all active segments, use the active incident cells and winner cells to modify synapses.
        for actCell in activeFCells:
            for segIndex in FCells[ actCell ].ReturnTerminalOn():
                NoRepeatInsort( toModSegs, segIndex )

        for segment in toModSegs:
            self.segments[ segment ].ModifyAllSynapses( FCells, activeFCells, lastActiveFCells )

#        for segment in self.segments:
#            if segment != None:
#                segment.ModifyAllSynapses( FCells, lastActiveFCells )

    def SegmentsAreConfident( self ):
    # Check the winner segments if they are confident, and return True if they all are.

# MIGHT WANT TO CHANGE THIS TO AN ABOVE THRESHOLD NUMBER OF WINNING SEGMENTS ARE CONFIDENT, SINCE I MIGHT RETURN TO WINNING SEGMENTS FOR EACH TERMINAL CELL.
        for winSeg in self.winnerSegments:
            if not self.segments[ winSeg ].IsConfident():
                return False

        return True

    def GetVector( self ):
    # returns the vector of appropriate size from the average of all winner segments.

        toReturn = []
        

    def StimulateSegments( self, FCells, activeCells, vectorSDR ):
    # Using the activeCells and next vector find all segments that activate from this.
    # Use these active segments to get the predicted cells for given column.

        predictedCells = []             # All the cells terminal on segments which become active due to vectorSDR and active incident cells.
        segsForTCells  = []             # The segs which are terminal on the predictedCells.
        potentialSegs  = []             # The segments that are stimulated and at zero vector.

        # First stimulate and activate any segments from the incident cells.
        # Activate the synapses in segments using activeCells.
        startTime = time()
        for incCell in activeCells:
            incCol, incSegments = FCells[ incCell ].ReturnIncidentOn()
            for entry in incSegments:
                self.segments[ entry ].IncidentCellActive( incCell, incCol )
        print( "   Incident Activation: " + str( time() - startTime ) )

        # Check the overlap of all segments and see which ones are active, and add the terminalCell to stimulatedCells.
        startTime = time()
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
        print( "   Check and Activate Segments: " + str( time() - startTime ) )

        # Choose the winner segment for each terminal cell.
        startTime = time()
        for entry in segsForTCells:
            highestConfidenceSeg   = entry[ 0 ]
            highestConfidenceValue = self.segments[ entry[ 0 ] ].ReturnConfidenceScore()

            for segIdx in entry:
                if self.segments[ segIdx ].ReturnConfidenceScore() > highestConfidenceValue:
                    highestConfidenceSeg   = segIdx
                    highestConfidenceValue = self.segments[ segIdx ].ReturnConfidenceScore()

            # Set this segment as the winner.
            self.segments[ highestConfidenceSeg ].SetAsWinner()
            NoRepeatInsort( self.winnerSegments, highestConfidenceSeg )
        print( "   Choose Winner Segment for each column: " + str( time() - startTime ) )

        return predictedCells
