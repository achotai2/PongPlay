from useful_functions import Within, IndexIfItsIn, NoRepeatInsort

class Classifier:

    def __init__( self, numCells, inputDimensions, radius ):
    # Initialize classifier.

        self.radius          = radius
        self.inputDimensions = inputDimensions

        # Keeps track of, for each cell, the indices of the states associated with it and the count.
        self.cellStateIndices = []
        self.cellStateCounts  = []
        for cell in range( numCells ):
            self.cellStateIndices.append( [] )
            self.cellStateCounts.append( [] )

        # Keeps track of each observed state.
        self.states = []

    def ClassifyState( self, inputState ):
    # Takes the input and finds a state in self.states that matches it.

# THIS MIGHT BE VERY SLOW. WE CAN SPEED IT UP A LOT BY SORTING THE STATE LIST BY DIM... THE PROBLEM THERE IS THAT WE NEED TO KEEP THE INDEX CONSTANT
        for stateIndex, state in enumerate( self.states ):
            match = True
            for dim in range( self.inputDimensions ):
                if not Within( inputState[ dim ], state[ dim ] - self.radius, state[ dim ] + self.radius, True ):
                    match = False
                    break

            if match:
                # Return the found stateIndex and stop loop.
                return stateIndex

        # If it doesn't find a match then create a new state.
        self.states.append( inputState.copy() )
        return len( self.states ) - 1

    def AddCount( self, activeCells, stateIndex ):
    # Add the count for given stateIndex to each cell in self.cells.

        for actCell in activeCells:
            index = NoRepeatInsort( self.cellStateIndices[ actCell ], stateIndex )

            if len( self.cellStateIndices[ actCell ] ) == len( self.cellStateCounts[ actCell ] ):
                # If found a reference then add to the count.
                self.cellStateCounts[ actCell ][ index ] += 1
            else:
                # If none found then create a new entry.
                self.cellStateCounts[ actCell ].insert( index, 1 )

    def Learn( self, activeCells, inputState ):
    # Take in the active cells and input state, add a count for that state to each active cell.

        if len( inputState ) != self.inputDimensions:
            print( "Classifier.Learn(): Input state received not of correct dimensions." )
            exit()

        # Get state index.
        stateIndex = self.ClassifyState( inputState )

        print( "Classifier Infer: " + str( len(self.states) ) )

        # Add to each cell count.
        self.AddCount( activeCells, stateIndex )

    def Infer( self, activeCells ):
    # Using given activeCells, present most probable state.

        states = []
        counts = []

        # Sum up all counts for all states associated with these activeCells.
        for actCell in activeCells:
            for item, stateIndex in enumerate( self.cellStateIndices[ actCell ] ):
                index = NoRepeatInsort( states, stateIndex )

                if len( states ) == len( counts ):
                    counts[ index ] += self.cellStateCounts[ actCell ][ item ]
                else:
                    # If none found then create a new entry.
                    counts.insert( index, self.cellStateCounts[ actCell ][ item ] )

        # Get total count.
        totalCount = 0
        for c in counts:
            totalCount += c

        # Calculate the state probabilities.
        probability = []
        for i, c in enumerate( counts ):
            probability.append( [ c * 100 / totalCount, self.states[ states[ i ] ] ] )

        probability.sort( key = lambda probability: probability[ 0 ], reverse = True )
        return probability
