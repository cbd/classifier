-module(classifier_utils).

-export([get_files/1, get_tokenized/2, get_tokenized/1, get_text_tokenized/1, count_tokens/1, 
        calculate_probabilities/5, get_ocurrences/2]).

-spec get_files(string()) -> list().
get_files(FolderName) ->
  SubDirs = [{Tag,filename:join([FolderName, Sub])} || {Sub,Tag} <- [{"neg", neg},{"pos", pos}]],
  Files = [[{Tag,File} || File <- filelib:wildcard(filename:join([Dir,"*.txt"]))] || {Tag,Dir} <- SubDirs],
  lists:foldl(fun(More, Accum) -> More ++ Accum end, [], Files).

-spec get_tokenized(string(), list()) -> [string()].
get_tokenized(Tag, Files) ->
  lists:flatmap(fun({FileTag, Filename}) when FileTag == Tag ->
    get_tokenized(Filename);
  ({_, _}) -> []
  end, Files).

-spec get_tokenized(string()) -> [string()].
get_tokenized(FileName) -> {ok, Data} = file:read_file(FileName), re:split(Data, "[^a-zA-Z0-9]+").

-spec get_text_tokenized(string()) -> [string()].
get_text_tokenized(Text) -> re:split(string:strip(Text), "[^a-zA-Z0-9]+").

-spec calculate_probabilities([string()], [string()], pos_integer(), float(), float()) -> dict().
calculate_probabilities(PosTokens, NegTokens, MinAppearances, MinProb, MaxProb) ->
  Tokens = lists:usort(PosTokens ++ NegTokens),
  PosTokenCounts = count_tokens(PosTokens),
  NegTokenCounts = count_tokens(NegTokens),
  LengthPosTokens = length(PosTokens),
  LengthNegTokens = length(NegTokens),

  lists:foldl(fun(Token, Dict) ->
    PosOcurrences = get_ocurrences(Token, PosTokenCounts),
    NegOcurrences = get_ocurrences(Token, NegTokenCounts), 

    case (PosOcurrences + NegOcurrences) < MinAppearances of
      true -> Dict;
      false ->
        % PosResult = min(1, 2 * PosOcurrences / LengthPosTokens),
        PosResult = try PosOcurrences / LengthPosTokens catch _:_ -> 0 end,
        NegResult = try NegOcurrences / LengthNegTokens catch _:_ -> 0 end,
        NegProbability = max(MinProb, min(MaxProb, NegResult / (PosResult + NegResult))),
        dict:store(Token, NegProbability, Dict)
    end
  end, dict:new(), Tokens).

count_tokens(Tokens) ->
  lists:foldl(fun(Token, Dict) -> 
    case dict:find(Token, Dict) of
      {ok, Count} -> dict:store(Token, Count+1, Dict);
      error -> dict:store(Token, 1, Dict)
    end 
  end, dict:new(), Tokens).

get_ocurrences(Token, TokenCounts) ->
  case dict:find(Token, TokenCounts) of
    {ok, Value} -> Value;
    error -> 0
  end.