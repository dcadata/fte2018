import datetime
import time

import pandas as pd


def _get_candidate_last_name(x: str) -> str:
    if pd.isna(x):
        return 'NA'
    x = x.split(' AND ')[0].split('/')[0]
    x = x.lower().strip()
    names = x.rsplit(None, 1)
    try:
        return names[1]
    except IndexError:
        return names[0]


def _add_margin(data: pd.DataFrame, merge_cols: list) -> pd.DataFrame:
    _separate_party = lambda p: data[data.party == p].drop(columns='party')
    data = _separate_party('D').merge(_separate_party('R'), on=merge_cols, suffixes=('D', 'R'))
    data['margin'] = (data.voteshareD - data.voteshareR).round(2)
    return data


def _read_and_normalize_2018_forecast(chamber: str) -> pd.DataFrame:
    """
    538's 2018 forecasts:
    Data links at bottom of page - search for "Download state data"
    Governor:
      Forecast: https://projects.fivethirtyeight.com/2018-midterm-election-forecast/governor/
      Data: https://projects.fivethirtyeight.com/congress-model-2018/governor_state_forecast.csv
    Senate:
      Forecast: https://projects.fivethirtyeight.com/2018-midterm-election-forecast/senate/
      Data: https://projects.fivethirtyeight.com/congress-model-2018/senate_seat_forecast.csv
    """
    data_filepath = dict(
        governor='2018_governor_state_forecast.csv',
        senate='2018_senate_seat_forecast.csv',
    )[chamber]
    fcst = pd.read_csv('data/' + data_filepath, usecols=[
        'forecastdate', 'state', 'special', 'party', 'candidate', 'voteshare', 'model'])

    fcst = fcst[fcst.model == 'classic'].drop(columns='model')  # 2018 forecasts defaulted to classic
    fcst.special = fcst.special.fillna(False)
    fcst.forecastdate = fcst.forecastdate.apply(lambda x: pd.to_datetime(x).date())
    # fcst['candidateLastName'] = fcst.candidate.apply(_get_candidate_last_name)
    return fcst


def _get_2018_forecast(*args) -> pd.DataFrame:
    fcst = _read_and_normalize_2018_forecast(*args)
    fcst = _add_margin(fcst, ['forecastdate', 'state', 'special'])
    return fcst


def _parse_gubernatorial_election_results_from_precinct_level_file() -> None:
    """
    From: https://github.com/MEDSL/2018-elections-official
    Via: https://electionlab.mit.edu/data
    """
    elex = pd.read_csv('data/STATE_precinct_general.csv', usecols=[
        'office', 'party_simplified', 'votes', 'candidate', 'special', 'state_po'])
    elex = elex[elex.office == 'GOVERNOR'].copy()
    elex = elex.groupby(['state_po', 'special', 'party_simplified', 'candidate'], as_index=False).votes.sum()
    total_votes = elex.groupby(['state_po', 'special'], as_index=False).votes.sum().rename(columns=dict(
        votes='totalvotes'))
    elex = elex.rename(columns=dict(votes='candidatevotes'))
    elex = elex.merge(total_votes, on=['state_po', 'special'])
    elex.to_csv('data/2018_governor_election_results.csv', index=False)


def _read_governor_election_results() -> pd.DataFrame:
    return pd.read_csv('data/2018_governor_election_results.csv')


def _read_senate_election_results() -> pd.DataFrame:
    """
    From: https://dataverse.harvard.edu/file.xhtml?fileId=4300300&version=5.0
    """
    elex = pd.read_csv('data/1976-2020_senate_election_results.csv', encoding='latin', usecols=[
        'year', 'state_po', 'stage', 'special', 'candidate', 'party_simplified',
        'candidatevotes', 'totalvotes',
    ])
    elex = elex[(elex.year == 2018) & (elex.stage == 'gen')].drop(columns=['year', 'stage'])
    return elex


def _add_voteshare_to_election_results_and_normalize_columns(elex: pd.DataFrame) -> pd.DataFrame:
    elex.party_simplified = elex.party_simplified.apply(lambda x: x[0])
    # elex['candidateLastName'] = elex.candidate.apply(_get_candidate_last_name)
    elex['voteshare'] = (elex.candidatevotes / elex.totalvotes).apply(lambda x: x * 100).round(2)
    elex = elex.drop(columns=['candidatevotes', 'totalvotes']).rename(columns=dict(
        state_po='state', party_simplified='party'))
    return elex


def _read_and_filter_election_results(chamber: str) -> pd.DataFrame:
    read_election_results_func = dict(
        governor=_read_governor_election_results,
        senate=_read_senate_election_results,
    )[chamber]
    elex = _add_voteshare_to_election_results_and_normalize_columns(read_election_results_func())
    return elex


def _get_election_results(*args) -> pd.DataFrame:
    elex = _read_and_filter_election_results(*args)
    elex = _add_margin(elex, ['state', 'special'])
    return elex


def _get_2022_forecast(chamber: str) -> pd.DataFrame:
    base_url = 'https://projects.fivethirtyeight.com/2022-general-election-forecast-data/'
    state_topline_filenames = dict(
        governor='governor_state_toplines_2022.csv',
        senate='senate_state_toplines_2022.csv',
    )
    data_filepath = base_url + state_topline_filenames[chamber]
    fcst = pd.read_csv(data_filepath, usecols=['district', 'expression', 'mean_netpartymargin'])
    # 2022 forecasts default to deluxe
    fcst = fcst[fcst.expression == '_deluxe'].drop_duplicates(subset=['district'], keep='first')
    fcst.mean_netpartymargin = fcst.mean_netpartymargin.round(2)
    fcst['seat'] = fcst.district.apply(lambda x: x[:2] if x.endswith(('-S3', '-G1')) else f'{x[:2]}-Special')
    fcst = fcst.drop(columns=['district', 'expression']).rename(columns=dict(mean_netpartymargin='marginFcst22'))
    return fcst


def _combine_forecast_and_election_results(chamber: str, use_today: bool = True, fcst_date: tuple = (2018, 11, 6)):
    fcst = _get_2018_forecast(chamber)
    cutoff_date = datetime.datetime.today().date() if use_today else datetime.date(*fcst_date)
    fcst = fcst[fcst.forecastdate == cutoff_date.replace(year=2018)].copy()

    fcst22 = _get_2022_forecast(chamber)

    elex = _get_election_results(chamber)

    combined = fcst.merge(elex, on=['state', 'special'], suffixes=('Fcst', 'Actl'))

    combined['seat'] = combined.state + combined.special.apply(lambda x: '-Special' if x else '')
    combined.forecastdate = combined.forecastdate.apply(lambda x: x.strftime('%m/%d/%Y'))
    combined['marginMiss'] = (combined.marginActl - combined.marginFcst).round(2)

    combined = combined.merge(fcst22, on='seat')
    combined['marginAdj22'] = combined.marginMiss + combined.marginFcst22

    return combined[[
        'forecastdate', 'seat',
        'voteshareDFcst', 'voteshareRFcst', 'candidateDFcst', 'candidateRFcst', 'marginFcst',
        'voteshareDActl', 'voteshareRActl', 'candidateDActl', 'candidateRActl', 'marginActl',
        'marginMiss', 'marginFcst22', 'marginAdj22',
    ]]


def main() -> None:
    params = {
        '1-Governors - model launch': dict(chamber='governor', use_today=False, fcst_date=(2022, 10, 11)),
        '2-Governors - closest to election': dict(chamber='governor', use_today=False),
        '3-Senate - this day in 2018': dict(chamber='senate'),
        '4-Senate - closest to election': dict(chamber='senate', use_today=False),
    }
    for label, i in params.items():
        _combine_forecast_and_election_results(**i).to_csv(f'outputs/{label}.csv', index=False)
        time.sleep(1)


if __name__ == '__main__':
    main()
