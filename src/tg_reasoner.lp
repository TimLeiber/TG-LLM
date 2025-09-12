% start(S, R, O, T): Event with subject S, relation R and object O starts at temporal index T where T is the number of months counting from year 0
start(S,R,O,T) :-
    event(S,R,O,SY,SM,_,_),
    T = SY*12 + (SM - 1).

% start(S, R, O, T): Event with subject S, relation R and object O ends at temporal index T where T is the number of months counting from year 0
end(S,R,O,T) :- 
    event(S,R,O,_,_,EY,EM),
    T = EY*12 + (EM - 1).

% event2idx(S, R, O, T): Event with subject S, relation R and object O starts at temporal index Idx where Idx ranges from 1 to n (interpretable order on events)
ordering_on_start_years(event(S, R, O), Idx + 1) :-
    start(S,R,O,T),
    Idx = #count {S2,R2,O2,T2 : start(S2,R2,O2,T2), T2 < T }.

% length(E, Y, M): event E has a duration of Y years and M months
length(event(S, R, O), Y, M) :-
    start(S,R,O,T),
    end(S,R,O,T'),
    L = T' - T,
    Y = L / 12,
    M = L \ 12.

% time_passed(E, E', Y, M): Event event E' started Y years and M months after event E'
time_passed(E, E', Y, M) :-
    start(S,R,O,T), E = event(S,R,O),
    start(S',R',O',T'), E' = event(S',R',O'),
    E != E',
    T >= T',
    L = T - T',
    Y = L / 12,
    M = L \ 12.

% starts_at(E, SY): event E has start year SY
starts_at(event(S, R, O), SY) :- event(S,R,O,SY,_,_,_).

% started_same_year(E, E'): event E and E' share the same start year
started_same_year(event(S, R, O), event(S', R', O')) :-
    event(S,R,O,SY,_,_,_),
    event(S',R',O',SY',_,_,_),
    SY == SY'.

% was_still_happening(E, E'):  Event E was still ongoing when event E' started
was_still_happening(E, E') :-
    start(S,R,O,T), end(S,R,O,EE), E  = event(S,R,O),
    start(S',R',O',T'), end(S',R',O',EE'), E' = event(S',R',O'),
    E != E',
    T <= T', T' <= EE.