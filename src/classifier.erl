-module(classifier).

-behaviour(gen_server).

-define(MINIMUM_APPEARANCES, 5).
-define(MIN_PROBABILITY, 0.01).
-define(MAX_PROBABILITY, 0.99).
-define(DEFAULT_PROBABILITY, 0.4).
-define(THRESHOLD_PROBABILITY, 0.9).
-define(MAX_TEXT_TOKENS, 5). 

-record(state, {
  token_probabilities :: dict(),
  neg_tokens = [] :: list(),
  pos_tokens = [] :: list(),
  new_neg_tokens = [] :: list(),
  new_pos_tokens = [] :: list()
  }).

-export([start_link/0, init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2, code_change/3]).
-export([state/0, train/1, tokens/0, classify/1, update_token_probabilities/0]).


-spec start_link() -> {ok, pid()}.
start_link() ->
  gen_server:start_link({local, ?MODULE}, ?MODULE,[],[]).

-spec train(string()) -> done.
train(Dir) ->
  gen_server:cast(?MODULE, {train, Dir}).

-spec tokens() -> list().
tokens() ->
  gen_server:call(?MODULE, tokens).

-spec classify(binary() | string()) -> float().
classify(Text) when is_binary(Text) ->
  classify(binary_to_list(Text));
classify(Text) ->
  gen_server:call(?MODULE, {classify, Text}).

-spec update_token_probabilities() -> ok.
update_token_probabilities() ->
  gen_server:cast(?MODULE, update_token_probabilities).

-spec state() -> #state{}.
state() ->
  gen_server:call(?MODULE, state).



%% ----------- %%

-spec init([]) -> {ok,#state{}}.
init([]) ->
  {ok, #state{token_probabilities = dict:new()}}.

-spec handle_call(term(), pid(), #state{}) -> {reply, term(), #state{}} | {noreply, #state{}}.
handle_call(state, _From, State) ->
  {reply, State, State};

handle_call(tokens, _From, State = #state{ token_probabilities=Tokens}) ->
  {reply, dict:to_list(Tokens), State};

handle_call({classify, Text}, _From, State = #state{token_probabilities=TokenProbabilities, new_pos_tokens=PosTokens, new_neg_tokens=NegTokens}) ->
  Tokens = re:split(string:strip(Text), "[^a-zA-Z0-9]+"),

  io:format("Tokens ~p~n",[Tokens]),

  Probalities = 
    lists:foldl(fun(Token, Accum) ->
      NewValue = 
        case dict:find(Token, TokenProbabilities) of
          {ok, Value} -> Value;
          error -> ?DEFAULT_PROBABILITY
        end,

      % case length(Accum) == round(length(Tokens)/2) of
      case length(Accum) == ?MAX_TEXT_TOKENS of
        false -> [NewValue | Accum] ;
        true ->
          lists:sublist(
          lists:sort(fun(A, B) ->
            abs(A - 0.5) >= abs(B - 0.5)
          end, [NewValue | Accum]), ?MAX_TEXT_TOKENS)
          % end, [NewValue | Accum]), round(length(Tokens)/2))
      end
    end, [], Tokens),

  io:format("Probalities ~p~n",[Probalities]),

  {NegMultiplication, PosMultiplication} = 
    lists:foldl(fun(P, {Neg, Pos}) ->
      {Neg*P, Pos*(1-P)}
    end,{1,1}, Probalities),

  io:format("NegMultiplication ~p~nPosMultiplication ~p ~n",[NegMultiplication, PosMultiplication]),

  TextProbability = NegMultiplication / (NegMultiplication + PosMultiplication),
  {TextStatus, NewPosTokens, NewNegTokens} =
    case TextProbability < ?THRESHOLD_PROBABILITY of
      true -> 
        {acceptable, lists:append(Tokens, PosTokens), NegTokens};
      false -> 
        {unacceptable, PosTokens, lists:append(Tokens, NegTokens)}
    end, 

  NewState = State#state{new_pos_tokens=NewPosTokens, new_neg_tokens=NewNegTokens},

  {reply, {TextStatus, TextProbability}, NewState};

handle_call(_Request, _From, State) ->
  {reply, ok, State}.

-spec handle_cast(term(), #state{}) -> {noreply, #state{}}.
handle_cast({train, Dir}, State = #state{new_pos_tokens=NewPosTokens, new_neg_tokens=NewNegTokens}) ->
  io:format("training...~n"),
  Files = get_files(Dir),
  
  PosTokens = lists:append(get_tokenized(pos, Files), NewPosTokens),
  NegTokens = lists:append(get_tokenized(neg, Files), NewNegTokens),

  TokenProbabilities = calculate_token_probabilities(PosTokens, NegTokens),

  NewState = State#state{token_probabilities = TokenProbabilities, pos_tokens=PosTokens, neg_tokens=NegTokens, new_pos_tokens=[], new_neg_tokens=[]},
  io:format("training done.~n"),
  {noreply, NewState};

handle_cast(update_token_probabilities, State = #state{new_pos_tokens=NewPosTokens, new_neg_tokens=NewNegTokens, pos_tokens=CurrentPosTokens, neg_tokens=CurrentNegTokens}) ->
  io:format("updating tokens...~n"),
  PosTokens = lists:append(NewPosTokens, CurrentPosTokens),
  NegTokens = lists:append(NewNegTokens, CurrentNegTokens),

  TokenProbabilities = calculate_token_probabilities(PosTokens, NegTokens),

  NewState = State#state{token_probabilities=TokenProbabilities, pos_tokens=PosTokens, neg_tokens=NegTokens, new_pos_tokens=[], new_neg_tokens=[]},
  io:format("updating done.~n"),
  {noreply, NewState};

handle_cast(Msg, State) ->
  io:format("bad message ~p",[Msg]),
  {noreply, State}.

-spec handle_info(term(), #state{}) -> {noreply, #state{}}.
handle_info(_Info, State) ->
  {noreply, State}.

-spec terminate(term(), #state{}) -> ok.
terminate(_Reason, _State) ->
  ok.

-spec code_change(term(), #state{}, term()) -> {ok, #state{}}.
code_change(_OldVsn, State, _Extra) ->
  {ok, State}.


get_files(FolderName) ->
  SubDirs = [{Tag,filename:join([FolderName, Sub])} || {Sub,Tag} <- [{"neg", neg},{"pos", pos}]],
  Files = [[{Tag,File} || File <- filelib:wildcard(filename:join([Dir,"*.txt"]))] || {Tag,Dir} <- SubDirs],
  lists:foldl(fun(More, Accum) -> More ++ Accum end, [], Files).

get_tokenized(Tag, Files) ->
  lists:flatmap(fun({FileTag, Filename}) when FileTag == Tag ->
    get_tokenized(Filename);
  ({_, _}) -> []
  end, Files).

get_tokenized(FileName) -> {ok, Data} = file:read_file(FileName), re:split(Data, "[^a-zA-Z0-9]+").

count_tokens(Tokens) ->
  lists:foldl(fun(Token, Dict) -> 
    case dict:find(Token, Dict) of
      {ok, Count} -> dict:store(Token, Count+1, Dict);
      error -> dict:store(Token, 1, Dict)
    end 
  end, dict:new(), Tokens).

calculate_token_probabilities(PosTokens, NegTokens) ->
  Tokens = lists:usort(PosTokens ++ NegTokens),
  PosTokenCounts = count_tokens(PosTokens),
  NegTokenCounts = count_tokens(NegTokens),
  LengthPosTokens = length(PosTokens),
  LengthNegTokens = length(NegTokens),

  lists:foldl(fun(Token, Dict) ->
    PosOcurrences = get_ocurrences(Token, PosTokenCounts),
    NegOcurrences = get_ocurrences(Token, NegTokenCounts), 

    case (PosOcurrences + NegOcurrences) < ?MINIMUM_APPEARANCES of
      true -> Dict;
      false ->
        % PosResult = min(1, 2 * PosOcurrences / LengthPosTokens),
        PosResult = try PosOcurrences / LengthPosTokens catch _:_ -> 0 end,
        NegResult = try NegOcurrences / LengthNegTokens catch _:_ -> 0 end,
        NegProbability = max(?MIN_PROBABILITY, min(?MAX_PROBABILITY, NegResult / (PosResult + NegResult))),
        dict:store(Token, NegProbability, Dict)
    end
  end, dict:new(), Tokens).

get_ocurrences(Token, TokenCounts) ->
  case dict:find(Token, TokenCounts) of
    {ok, Value} -> Value;
    error -> 0
  end.