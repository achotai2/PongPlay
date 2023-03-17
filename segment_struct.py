from random import uniform, choice, sample, randrange, shuffle
from bisect import bisect_left
from useful_functions import SmoothStep, BinarySearch, NoRepeatInsort, ModThisSynapse, IndexIfItsIn, FastIntersect, DelIfIn, IndexOfGreatest, NoRepeatConcatenate
import numpy
import math
from time import time

class Segment:

    def __init__( self, vectorMemoryDict, ID, incidentColumns, terminalColumns, incidentCells, terminalCells, vectorSDR, positionSDR ):
    # Initialize the inidividual segment structure class.
    # Generate random permanence connections to all received incident cells, and to terminal cell.

        self.ID = ID

        self.vectorMemoryDict = vectorMemoryDict

        self.active              = True                        # True means it's predictive, and above threshold terminal cells fired.
        self.winner              = True                       # Only the segment terminal on cell which has highest confidenceScore becomes winner.

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
        self.vectorSynapses  = vectorSDR.copy()
        self.vectorPermanences = []
        for iCell in vectorSDR:
            self.vectorPermanences.append( uniform( 0, vectorMemoryDict[ "initialPermanence" ] ) )
        self.vectorThreshold = vectorMemoryDict[ "FActivationThresholdMin" ]   # Minimum overlap required to activate segment.

        # Position portion.
        self.positionSynapses  = positionSDR.copy()
        self.positionPermanences = []
        for iCell in positionSDR:
            self.positionPermanences.append( uniform( 0, vectorMemoryDict[ "initialPermanence" ] ) )
        self.positionThreshold = vectorMemoryDict[ "FActivationThresholdMin" ]   # Minimum overlap required to activate segment.

    def Inside( self, vectorSDR ):
    # Checks if given vector position fits with this segment.

        # Compute the overlap with the segments synapses.
        intersection = FastIntersect( vectorSDR, self.vectorSynapses )

        if len( intersection ) >= self.vectorThreshold:
            return True
        else:
            return False

# CAN MAKE BOTH ABOVE AND BELOW ONE FUNCTION, WITH MULTIPLE ALLOWABLE SECONDARY INPUT STREAMS (MAYBE EVEN PRIMARY TOO)

    def CheckPosition( self, positionSDR ):
    # Checks if given position fits with this segment

        # Compute the overlap with the segments synapses.
        intersection = FastIntersect( positionSDR, self.positionSynapses )

# SHOULD MAKE IT SO THAT THE permanenceLowerThreshold COMES IN TO PLAY HERE.

        if len( intersection ) >= self.positionThreshold:
            return True
        else:
            return False

    def CheckTerminalActivation( self, activeTerminalColumns ):
    # Check the overlap of terminal columns and activeTerminalColumns.

        intersection = FastIntersect( self.terminalColumns, activeTerminalColumns )

# SHOULD MAKE IT SO THAT THE permanenceLowerThreshold COMES IN TO PLAY HERE.

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

    def RemoveTerminalSynapse( self, Cells, cellToDelete ):
    # Delete the terminal synapse sent.

        Cells[ cellToDelete ].DeleteTerminalSegmentReference( self.ID )

        cellIndexToDelete = IndexIfItsIn( self.terminalSynapses, cellToDelete )
        del self.terminalSynapses[ cellIndexToDelete ]
        del self.terminalColumns[ cellIndexToDelete ]
        del self.terminalPermanences[ cellIndexToDelete ]

        if len( self.terminalSynapses ) == 0:
            self.markedForDeletion = True

    def NewTerminalSynapse( self, Cells, synapseToCreate ):
    # Create a new incident synapse.

        Cells[ synapseToCreate ].TerminalToThisSeg( self.ID )

        synLengthBefore = len( self.terminalSynapses )
        colLengthBefore = len( self.terminalColumns )

        indexToPut = NoRepeatInsort( self.terminalSynapses, synapseToCreate )
        self.terminalPermanences.insert( indexToPut, uniform( 0, self.vectorMemoryDict[ "initialPermanence" ] ) )
        NoRepeatInsort( self.terminalColumns, Cells[ synapseToCreate ].column )

        if len( self.terminalSynapses ) == synLengthBefore:
            print( "NewIncidentSynapse(): Tried to add an incident synapse, but already exists." )
            exit()
        if len( self.terminalColumns ) == colLengthBefore:
            print( "NewIncidentSynapse(): Tried to add an incident synapse, but synapse to this column already exists." )
            exit()

    def RemoveIncidentSynapse( self, Cells, cellToDelete ):
    # Delete the incident synapse sent.

        Cells[ cellToDelete ].DeleteIncidentSegmentReference( self.ID )

        cellIndexToDelete = IndexIfItsIn( self.incidentSynapses, cellToDelete )
        del self.incidentSynapses[ cellIndexToDelete ]
        del self.incidentPermanences[ cellIndexToDelete ]
        del self.incidentColumns[ cellIndexToDelete ]

        if len( self.incidentSynapses ) <= self.vectorMemoryDict[ "FActivationThresholdMin" ]:
            self.markedForDeletion = True

    def NewIncidentSynapse( self, Cells, synapseToCreate ):
    # Create a new incident synapse.

        Cells[ synapseToCreate ].IncidentToThisSeg( self.ID )

        synLengthBefore = len( self.incidentSynapses )
        colLengthBefore = len( self.incidentColumns )

        indexToPut = NoRepeatInsort( self.incidentSynapses, synapseToCreate )
        self.incidentPermanences.insert( indexToPut, uniform( 0, self.vectorMemoryDict[ "initialPermanence" ] ) )
        NoRepeatInsort( self.incidentColumns, Cells[ synapseToCreate ].column )

        if len( self.incidentSynapses ) == synLengthBefore:
            print( "NewIncidentSynapse(): Tried to add an incident synapse, but already exists." )
            exit()
        if len( self.incidentColumns ) == colLengthBefore:
            print( "NewIncidentSynapse(): Tried to add an incident synapse, but synapse to this column already exists." )
            exit()

    def ModifyTransformSynapses( self, vectorLastActive ):
    # 1.) Increase permanence strength to last-active incident cells that already have synapses;
    # 2.) Decrease synapse strength to inactive incident cells that already have synapses;
    # 3.) Build new synapses to active incident winner cells that don't have synapses;

        incSynapseToAdd = vectorLastActive.copy()                      # Used for 3 below.

        for incCell in self.vectorSynapses:
            # 1.)...
            incIndex = IndexIfItsIn( self.vectorSynapses, incCell )

            if BinarySearch( vectorLastActive, incCell ):
                self.vectorPermanences[ incIndex ] = ModThisSynapse( self.vectorPermanences[ incIndex ], self.vectorMemoryDict[ "permanenceIncrement" ], 0.0, 1.0 )
                DelIfIn( incSynapseToAdd, incCell )
            # 2.)...
            else:
                self.vectorPermanences[ incIndex ] = ModThisSynapse( self.vectorPermanences[ incIndex ], -self.vectorMemoryDict[ "permanenceDecrement" ], 0.0, 1.0 )
                # If it is below 0.0 then delete it.
                if self.vectorPermanences[ incIndex ] <= 0.0:
                    del self.vectorSynapses[ incIndex ]
                    del self.vectorPermanences[ incIndex ]

                    if len( self.vectorSynapses ) <= self.vectorMemoryDict[ "FActivationThresholdMin" ]:
                        self.markedForDeletion = True
        # 3.)...
        if len( incSynapseToAdd ) > 0:
            # Choose which synapses to actually add, since we can only add n-number per time step.
            reallyRealSynapsesToAdd = sample( incSynapseToAdd, min( len( incSynapseToAdd ), self.vectorMemoryDict[ "maxSynapsesToAddPer" ] ) )
            for toAdd in reallyRealSynapsesToAdd:
                synLengthBefore = len( self.vectorSynapses )

                indexToPut = NoRepeatInsort( self.vectorSynapses, toAdd )
                self.vectorPermanences.insert( indexToPut, uniform( 0, self.vectorMemoryDict[ "initialPermanence" ] ) )

                if len( self.vectorSynapses ) == synLengthBefore:
                    print( "ModifyTransformSynapses(): Tried to add an incident synapse, but already exists." )
                    exit()

            # If the number of synapses is above maxSynapsesPerSegment then delete random synapses.
# PROBABLY BETTER TO REMOVE THE ONE WITH THE LOWEST PERMANENCE, THIS SHOULDNT BE HARD TO DO EITHER AS THERE IS A FUNCTION TO DO THIS.
            while self.vectorMemoryDict[ "maxSynapsesPerSegment" ] - len( self.vectorSynapses ) < 0:
                toDelIndex = randrange( len( self.vectorSynapses ) )
                del self.vectorSynapses[ toDelIndex ]
                del self.vectorPermanences[ toDelIndex ]

                if len( self.vectorSynapses ) <= self.vectorMemoryDict[ "FActivationThresholdMin" ]:
                    self.markedForDeletion = True

    def ModifyPositionSynapses( self, positionLastActive ):
    # 1.) Increase permanence strength to last-active incident cells that already have synapses;
    # 2.) Decrease synapse strength to inactive incident cells that already have synapses;
    # 3.) Build new synapses to active incident winner cells that don't have synapses;

        incSynapseToAdd = positionLastActive.copy()                      # Used for 3 below.

        print(incSynapseToAdd)

        for incCell in self.positionSynapses:
            # 1.)...
            incIndex = IndexIfItsIn( self.positionSynapses, incCell )

            if BinarySearch( positionLastActive, incCell ):
                self.positionPermanences[ incIndex ] = ModThisSynapse( self.positionPermanences[ incIndex ], self.vectorMemoryDict[ "permanenceIncrement" ], 0.0, 1.0 )
                DelIfIn( incSynapseToAdd, incCell )
            # 2.)...
            else:
                self.positionPermanences[ incIndex ] = ModThisSynapse( self.positionPermanences[ incIndex ], -self.vectorMemoryDict[ "permanenceDecrement" ], 0.0, 1.0 )
                # If it is below 0.0 then delete it.
                if self.positionPermanences[ incIndex ] <= 0.0:
                    del self.positionSynapses[ incIndex ]
                    del self.positionPermanences[ incIndex ]

                    if len( self.positionSynapses ) <= self.vectorMemoryDict[ "FActivationThresholdMin" ]:
                        self.markedForDeletion = True
        # 3.)...
        if len( incSynapseToAdd ) > 0:
            print(incSynapseToAdd)

            # Choose which synapses to actually add, since we can only add n-number per time step.
            reallyRealSynapsesToAdd = sample( incSynapseToAdd, min( len( incSynapseToAdd ), self.vectorMemoryDict[ "maxSynapsesToAddPer" ] ) )
            for toAdd in reallyRealSynapsesToAdd:
                synLengthBefore = len( self.positionSynapses )

                print( self.positionSynapses)
                print(toAdd)

                indexToPut = NoRepeatInsort( self.positionSynapses, toAdd )
                self.positionPermanences.insert( indexToPut, uniform( 0, self.vectorMemoryDict[ "initialPermanence" ] ) )

                if len( self.positionSynapses ) == synLengthBefore:
                    print( "ModifyPositionSynapses(): Tried to add an incident synapse, but already exists." )
                    exit()

            # If the number of synapses is above maxSynapsesPerSegment then delete random synapses.
# PROBABLY BETTER TO REMOVE THE ONE WITH THE LOWEST PERMANENCE, THIS SHOULDNT BE HARD TO DO EITHER AS THERE IS A FUNCTION TO DO THIS.
            while self.vectorMemoryDict[ "maxSynapsesPerSegment" ] - len( self.positionSynapses ) < 0:
                toDelIndex = randrange( len( self.positionSynapses ) )
                del self.positionSynapses[ toDelIndex ]
                del self.positionPermanences[ toDelIndex ]

                if len( self.positionSynapses ) <= self.vectorMemoryDict[ "FActivationThresholdMin" ]:
                    self.markedForDeletion = True

    def ModifyAllPrimaryIncidentSynapses( self, Cells, lastActive ):
    # 1.) Increase permanence strength to last-active incident cells that already have synapses;
    # 2.) Decrease synapse strength to inactive incident cells that already have synapses;
    # 3.) Build new synapses to active incident winner cells that don't have synapses;

        incSynapseToAdd = lastActive.copy()                      # Used for 3 below.

        for incCell in self.incidentSynapses:
            # 1.)...
            incIndex = IndexIfItsIn( self.incidentSynapses, incCell )

            if Cells[ incCell ].lastActive:
                self.incidentPermanences[ incIndex ] = ModThisSynapse( self.incidentPermanences[ incIndex ], self.vectorMemoryDict[ "permanenceIncrement" ], 0.0, 1.0 )
                DelIfIn( incSynapseToAdd, incCell )
            # 2.)...
            else:
                self.incidentPermanences[ incIndex ] = ModThisSynapse( self.incidentPermanences[ incIndex ], -self.vectorMemoryDict[ "permanenceDecrement" ], 0.0, 1.0 )
                # If it is below 0.0 then delete it.
                if self.incidentPermanences[ incIndex ] <= 0.0:
                    self.RemoveIncidentSynapse( Cells, incCell )

        # 3.)...
        if len( incSynapseToAdd ) > 0:
            # Check to make sure this segment doesn't already have a synapse to this column.
            realSynapsesToAdd = []

            for synAdd in incSynapseToAdd:
                if Cells[ synAdd ].lastWinner and not BinarySearch( self.incidentColumns, Cells[ synAdd ].column ):
                    realSynapsesToAdd.append( synAdd )

            # Choose which synapses to actually add, since we can only add n-number per time step.
            reallyRealSynapsesToAdd = sample( realSynapsesToAdd, min( len( realSynapsesToAdd ), self.vectorMemoryDict[ "maxSynapsesToAddPer" ] ) )
            for toAdd in reallyRealSynapsesToAdd:
                self.NewIncidentSynapse( Cells, toAdd )

            # If the number of synapses is above maxSynapsesPerSegment then delete random synapses.
# PROBABLY BETTER TO REMOVE THE ONE WITH THE LOWEST PERMANENCE, THIS SHOULDNT BE HARD TO DO EITHER AS THERE IS A FUNCTION TO DO THIS.
            while self.vectorMemoryDict[ "maxSynapsesPerSegment" ] - len( self.incidentSynapses ) < 0:
                toDelIndex = randrange( len( self.incidentSynapses ) )
                self.RemoveIncidentSynapse( Cells, self.incidentSynapses[ toDelIndex ] )

    def ModifyAllTerminalSynapses( self, Cells, active ):
    # 1.) Decrease all terminal permanences to currently non-winner cells.
    # 2.) If terminal cell is winner increase permance for this terminal synapse;
    # 3.) Build new synapses to active terminal winner cells that don't have synapses;

        terSynapseToAdd = active.copy()                          # Used for 6 below.

        # 1.)...
        for terCell in self.terminalSynapses:
            terIndex = IndexIfItsIn( self.terminalSynapses, terCell )

            if not Cells[ terCell ].active:
                self.terminalPermanences[ terIndex ] = ModThisSynapse( self.terminalPermanences[ terIndex ], -self.vectorMemoryDict[ "permanenceDecrement" ], 0.0, 1.0 )
                if self.terminalPermanences[ terIndex ] <= 0.0:
                    # If it is below 0.0 then delete it.
                    self.RemoveTerminalSynapse( Cells, terCell )
        # 2.)...
            else:
                self.terminalPermanences[ terIndex ] = ModThisSynapse( self.terminalPermanences[ terIndex ], self.vectorMemoryDict[ "permanenceIncrement" ], 0.0, 1.0 )

            if Cells[ terCell ].active:
                DelIfIn( terSynapseToAdd, terCell )

        # 3.)...
        if len( terSynapseToAdd ) > 0:
            # Check to make sure this segment doesn't already have a synapse to this column.
            realSynapsesToAdd = []

            for synAdd in terSynapseToAdd:
                if Cells[ synAdd ].winner and not BinarySearch( self.terminalColumns, Cells[ synAdd ].column ):
                    realSynapsesToAdd.append( synAdd )

            # Choose which synapses to actually add, since we can only add n-number per time step.
            reallyRealSynapsesToAdd = sample( realSynapsesToAdd, min( len( realSynapsesToAdd ), self.vectorMemoryDict[ "maxSynapsesToAddPer" ] ) )
            for toAdd in reallyRealSynapsesToAdd:
                self.NewTerminalSynapse( Cells, toAdd )

            # If the number of synapses is above maxSynapsesPerSegment then delete random synapses.
# PROBABLY BETTER TO REMOVE THE ONE WITH THE LOWEST PERMANENCE, THIS SHOULDNT BE HARD TO DO EITHER AS THERE IS A FUNCTION TO DO THIS.
            while self.vectorMemoryDict[ "maxSynapsesPerSegment" ] - len( self.incidentSynapses ) < 0:
                toDelIndex = randrange( len( self.terminalSynapses ) )
                self.RemoveTerminalSynapse( Cells, self.incidentSynapses[ toDelIndex ] )

    def SynapseLearning( self, Cells, active, lastActive, vectorLastActive, positionLastActive ):
    # Perform learning on all synapses based on cell and column activations.

        if self.winner:
            self.ModifyAllPrimaryIncidentSynapses( Cells, lastActive )
            self.ModifyTransformSynapses( vectorLastActive )
            self.ModifyPositionSynapses( positionLastActive )
            self.ModifyAllTerminalSynapses( Cells, active )
            self.UpdateConfidenceScore()
        else:
        # Also Decay terminal synapses if it is not active but the terminal cell is active.
#            if not self.IsConfident():
            for terCell in self.terminalSynapses:
                terIndex = IndexIfItsIn( self.terminalSynapses, terCell )
                self.terminalPermanences[ terIndex ] = ModThisSynapse( self.terminalPermanences[ terIndex ], -self.vectorMemoryDict[ "permanenceDecay" ], 0.0, 1.0 )
                # If it is below 0.0 then delete it.
                if self.terminalPermanences[ terIndex ] <= 0.0:
                    self.RemoveTerminalSynapse( Cells, terCell )

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

    def CheckIncidentActivation( self ):
    # Checks the incidentActivation against activationThreshold to see if segment becomes active and stimulated.

        if len( self.incidentActivation ) >= self.activationThreshold:
            self.active            = True
            return True

        self.active = False
        return False

    def SetAsWinner( self ):
    # Set as winner.

        self.winner = True

    def GetTerminalSynapses( self ):
    # Return the terminal synapses and their permanence strengths.

        return self.terminalSynapses.copy(), self.terminalPermanences.copy(), self.terminalColumns.copy()

    def GetValidTerminalSynapses( self ):
    # Return only the terminal synapses if they are above lowest threshold.

        valid = []
        for i, cell in enumerate( self.terminalSynapses ):
            if self.terminalPermanences[ i ] >= self.vectorMemoryDict[ "permanenceLowerThreshold" ]:
                valid.append( cell )

        return valid

    def GetIncidentSynapses( self ):
    # Return the terminal synapses and their permanence strengths.

        return self.incidentSynapses.copy(), self.incidentPermanences.copy()

    def ReturnTransformSDR( self ):
    # Returns the transformation SDR.

        return self.vectorSynapses.copy()

    def ReturnTerminalActivation( self ):
    # Returns the sum of terminal permanences.

        sum = 0.0
        for permanence in self.terminalPermanences:
            sum += permanence

        return sum

    def ReturnIncidentActivation( self ):
    # Return the amount of principal incident activation.

# DO I WANT TO RETURN A SUM OF ALL ACTIVATION OVERLAP, NOT JUST INCIDENT?
        return len( self.incidentActivation )

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
        self.activationList     = []

    def HowManyActiveSegs( self ):
    # Return the number of active segments.

        return len( self.activeSegments )

    def HowManySegs( self ):
    # Return the number of segments.

        return self.vectorMemoryDict[ "maxTotalSegments" ] - len( self.availableSegments )

    def HowManyWinnerSegs( self ):
    # Return the number of winner segments.

        return len( self.winnerSegments )

    def DeleteSegmentsAndSynapse( self, Cells, segIndex ):
    # Goes through all segments and checks if they are marked for deletion. Deletes these segments from self.segments,
    # and removes all references to them in self.incidentSegments.

        if self.segments[ segIndex ] != None:
            # Delete any references to segment in Cells.
            incidentCells = self.segments[ segIndex ].GetIncidentSynapses()[ 0 ]
            for iCell in incidentCells:
                Cells[ iCell ].DeleteIncidentSegmentReference( segIndex )
            terminalCells = self.segments[ segIndex ].GetTerminalSynapses()[ 0 ]
            for tCell in terminalCells:
                Cells[ tCell ].DeleteTerminalSegmentReference( segIndex )

            # Delete any references to this segment if they exist in activeSegments.
            DelIfIn( self.activeSegments, segIndex )

            # Add entry index available again.
            NoRepeatInsort( self.availableSegments, segIndex )

            # Delete the segment by setting the entry to None state.
            self.segments[ segIndex ] = None

    def CreateSegment( self, Cells, incidentColumns, terminalColumns, incidentCells, terminalCells, vectorSDR, positionSDR ):
    # Creates a new segment.

        # If no segments available then randomly delete one below confidene threshold.
        while len( self.availableSegments ) == 0:
            checkIdx = randrange( self.vectorMemoryDict[ "maxTotalSegments" ] )
            if not self.segments[ checkIdx ].IsConfident():
                self.DeleteSegmentsAndSynapse( Cells, checkIdx )
#            print( "CreateSegment(): Want to create new segment, but none left available." )
#            exit()

        # Get index and remove it from list.
        indexOfNew = self.availableSegments.pop( 0 )

        # Assemble and generate the new segment, and add it to list.
        newSegment = Segment( self.vectorMemoryDict, indexOfNew, incidentColumns, terminalColumns, incidentCells, terminalCells, vectorSDR, positionSDR )
        if self.segments[ indexOfNew ] == None:
            self.segments[ indexOfNew ] = newSegment
        else:
            print( "CreateSegment(): Tried to create new segment but this entry was already full." )
            exit()

        # Add segment to active segments list and winners list.
        NoRepeatInsort( self.activeSegments, indexOfNew )
        NoRepeatInsort( self.winnerSegments, indexOfNew )

        # Add reference to segment in incident cells.
        for incCell in incidentCells:
            # Add segment but also check if incident cell has too many segments attached.
            if Cells[ incCell ].IncidentToThisSeg( indexOfNew ):
                # If if does then remove the segment with the lowest confidence.
                asIncident       = Cells[ incCell ].ReturnIncidentOn()
                lowestConfidence = 0.0
                lowestSeg        = asIncident[ 0 ]
                for seg in asIncident:
                    if self.segments[ seg ].ReturnConfidenceScore() < lowestConfidence:
                        lowestConfidence = self.segments[ seg ].ReturnConfidenceScore()
                        lowestSeg        = seg

                # Remove reference from both segment to Cell, and Cell to segment.
                self.segments[ lowestSeg ].RemoveIncidentSynapse( Cells, incCell )

        # Add reference to segment in terminal cells.
        for tCell in terminalCells:
            Cells[ tCell ].TerminalToThisSeg( indexOfNew )

    def UpdateSegmentActivity( self, Cells ):
    # Make every segment that was active inactive, and refreshes its synapse activation.
    # Also add a time step to each segment, and see if it dies as a result. Delete any segments that die.
    # Also refresh Cells terminal activation.

        toDelete = []

        # Refresh state of all active and winner segments.
        for actSeg in self.activeSegments:
            if self.segments[ actSeg ].RefreshSegment():
                toDelete.append( actSeg )

        # Delete segments with no terminal or incident synapses.
        for delSeg in toDelete:
            self.DeleteSegmentsAndSynapse( Cells, delSeg )

        self.activeSegments = []
        self.winnerSegments = []
        self.activationList = []

    def SegmentLearning( self, Cells, activeCells, lastActiveCells, lastVectorSDR, lastPositionSDR ):
    # Perform learning on all active and inactive segments.
    # Refresh all segments then perform learning on them.
    # Delete segments that need deleting.

        for actSeg in self.activeSegments:
            self.segments[ actSeg ].SynapseLearning( Cells, lastActiveCells, activeCells, lastVectorSDR, lastPositionSDR )

    def ChooseVectorSegment( self ):
    # Go through all active stimulated segments and check their transformationSDRs.
    # Form a transformSDR from the most confident segments which fits.
    # Returns the vector of appropriate size from the average of all winner segments, and the predicted cells.

# WE WILL USE THE FACT THAT SEGMENTS BRANCH TERMINAL CELLS TO SIMPLIFY, BUT LATER SHOULD EXTEND THIS TO MORE GENERAL IF NOT BRANCHING.
        # Find the active segment with the highest confidence.
        index = IndexOfGreatest( [ self.segments[ i ].ReturnConfidenceScore() for i in self.activeSegments ] )

        if len( self.activeSegments ) > 0 and self.segments[ self.activeSegments[ index ] ].IsConfident():
            # If the greatest confidence segment is above threshold confidence then return its terminal cells and its vector.
            terminalSynapses = self.segments[ self.activeSegments[ index ] ].GetTerminalSynapses() [ 0 ]
            vectorSDR        = self.segments[ self.activeSegments[ index ] ].ReturnTransformSDR()

            return terminalSynapses, vectorSDR

        else:
            return [], []

    def ChooseWinnerSegment( self, activeColumns, lastVectorSDR, lastPositionSDR ):
    # Use the active segments from last time step, plus the last motor vectorSDR, plus the presently active columns, to choose a winner segment(s).

# SINCE THIS IS SOMETHING SPECIFIC TO VECTOR MEMORY, TO MAKE IT MORE GENERAL WE COULD MOVE THIS TO VECTOR MEMORY.

        validSegments = []

        # Check the active segments terminal against activeColumns for overlap.
        for actSeg in self.activeSegments:
            if self.segments[ actSeg ].CheckTerminalActivation( activeColumns ) and self.segments[ actSeg ].Inside( lastVectorSDR ) and self.segments[ actSeg ].CheckPosition( lastPositionSDR ):
                validSegments.append( actSeg )

        # Choose the valid segment with the greatest confidence, even if not above threshold confidence. If there are no valid segments return None.
        winIndex = IndexOfGreatest( [ self.segments[ i ].ReturnConfidenceScore() for i in validSegments ] )

        if winIndex != None:
            # Make this segment the winner.
            self.segments[ validSegments[ winIndex ] ].SetAsWinner()
            NoRepeatInsort( self.winnerSegments, validSegments[ winIndex ] )
            return True

        else:
            return False

    def StimulateSegments( self, Cells, activeCells ):
    # Using the activeCells and next vector find all segments that activate from this.
    # Use these active segments to get the predicted cells for given column.

        # First stimulate and activate any segments from the incident cells.
        # Activate the synapses in segments using activeCells.
        for incCell in activeCells:
            incCol      = Cells[ incCell ].ReturnColumn()
            incSegments = Cells[ incCell ].ReturnIncidentOn()
            for entry in incSegments:
                self.segments[ entry ].IncidentCellActive( incCell, incCol )

        activationList = []

        # Check the overlap of all segments and see which ones are active, and add the terminalCell to stimulatedCells.
        for segment in self.segments:
            if segment != None and segment.CheckIncidentActivation():                   # Checks incident overlap above threshold and if vector is inside.
                NoRepeatInsort( self.activeSegments, segment.ID )

# RIGHT NOW WE ONLY CHECK PRIMARY INCIDENCE TO MAKE SEGMENTS ACTIVE. TO MAKE MORE GENERAL WE SHOULD ALSO CHECK SECONDARY INCIDENCE, BUT CAN ALSO PASS NONE
# IF IT DOES'T MATTER (EX. SPATIAL POOLER).

# SHOULD MOVE A LOT OF THIS VECTOR BASED STUFF ACTUALLY TO VECTOR MEMORY AND LET segment_struct HANDLE SEGMENTS MORE GENERALLY.

            # Order the segments by activation and put these indices in activationList.
            activation  = segment.ReturnIncidentActivation()
            insertIndex = RepeatInsort( activationList, activation )
            self.activationList.insert( insertIndex, segment.ID )

        # Write the order of the segments from highest overlap to least overlap.
        self.activationList.reverse()

    def ReturnWinnerCells( self ):
    # Return the cells of the winner segments.

        winnerCells = []
        winnerCols  = []

        for winSeg in self.winnerSegments:
            winnerCells = NoRepeatConcatenate( winnerCells, self.segments[ winSeg ].GetTerminalSynapses()[ 0 ] )
            winnerCols  = NoRepeatConcatenate( winnerCols, self.segments[ winSeg ].GetTerminalSynapses()[ 2 ] )

# DONT NEED TO CHECK IF ANY OF THE WINNER CELLS CHOSEN ARE OF THE SAME COLUMN NOW, BECAUSE WE SHOULD ONLY HAVE ONE WINNER SEGMENT, BUT IF WE HAVE
# NOT BRANCHING TERMINAL THEN WE WILL NEED TO CHECK THIS.

        return winnerCells, winnerCols

    def ReturnCellsFromStimulatedSegs( self, numberofActiveSegs ):
    # Return the terminal cells of all active segments as a list, at most numberofActiveSegs, the ones with the highest overlap of primary incidence.

        count        = 0
        terminalList = []

        # Go through the list of segments, ordered from highest overlapscore to least.
        while count < len( self.activationList ) and count < numberofActiveSegs:
            terminalList.append( self.segments[ self.activationList[ count ] ].GetValidTerminalSynapses() )
            count += 1

        return terminalList

    def ReturnHighestOverlapActiveSegment( self ):
    # Returns the terminal cells of the highest overlapping incident primary segment, which is also active.

        if self.segments[ self.activationList[ 0 ] ].active:
            return self.segments[ self.activationList[ 0 ] ].GetValidTerminalSynapses()
        else:
            return []
