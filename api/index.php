<?php
/* ccminer API sample UI (API 1.9) */

// Properly retreive URL that has been called with full
// scheme, port, reverse proxy support...
// https://stackoverflow.com/a/8891890/8998305
include "url_origin.inc.php";
$origin_url = full_url($_SERVER, true);

// Strip "/index.php" from the URL
// https://stackoverflow.com/a/25850242/8998305
$base_url = explode('/', $origin_url);
array_pop($base_url);
$base_url = implode('/', $base_url) . '/'; 

$configs = array(
	'LOCAL'=>'local-sample.php',
	//'EPSYTOUR'=>'epsytour.php', /* copy local.php file and edit target IP:PORT */
);

// 3 seconds max.
set_time_limit(3);
error_reporting(0);

function getdataFromPeers()
{
	global $base_url, $configs;
	$data = array();
	foreach ($configs as $name => $conf) {
		print("<p>$base_url"."$conf</p>");

		$json = file_get_contents($base_url.$conf);

		$data[$name] = json_decode($json, TRUE);
	}
	return $data;
}

function ignoreField($key)
{
	$ignored = array(
		'API','VER','GPU','BUS','POOLS',
		'CARD','GPUS','CPU','TS','URL',
	);
	return in_array($key, $ignored);
}

function translateField($key)
{
	$intl = array();
	$intl['NAME'] = 'Software';
	$intl['VER'] = 'Version';

	$intl['ALGO'] = 'Algorithm';
	$intl['GPUS'] = 'GPUs';
	$intl['CPUS'] = 'Threads';
	$intl['KHS'] = 'Hash rate';
	$intl['ACC'] = 'Accepted shares';
	$intl['ACCMN'] = 'Accepted / mn';
	$intl['REJ'] = 'Rejected';
	$intl['SOLV'] = 'Solved';
	$intl['BEST'] = 'Best share';
	$intl['STALE'] = 'Stale shares';
	$intl['LAST'] = 'Last share';
	$intl['DIFF'] = 'Difficulty';
	$intl['NETKHS'] = 'Net Rate';
	$intl['UPTIME'] = 'Miner up time';
	$intl['TS'] = 'Last update';
	$intl['THR'] = 'Throughput';
	$intl['WAIT'] = 'Wait time';

	$intl['H'] = 'Bloc height';
	$intl['I'] = 'Intensity';
	$intl['HWF'] = 'Failures';
	$intl['POOL'] = 'Pool';
	$intl['POOLS'] = 'Pools';

	$intl['TEMP'] = 'T°c';
	$intl['FAN'] = 'Fan %';
	$intl['CPUFREQ'] = 'CPU Freq.';
	$intl['FREQ'] = 'Base Freq.';
	$intl['MEMFREQ'] = 'Mem. Freq.';
	$intl['GPUF'] = 'Curr Freq.';
	$intl['MEMF'] = 'Mem. Freq.';
	$intl['KHW'] = 'Efficiency';
	$intl['POWER'] = 'Power';
	$intl['PLIM'] = 'P.Limit';
	$intl['PST'] = 'P-State';

	// pool infos
	$intl['POOL'] = 'Pool';
	$intl['PING'] = 'Ping (ms)';
	$intl['DISCO'] = 'Disconnects';
	$intl['USER'] = 'User';

	if (isset($intl[$key]))
		return $intl[$key];
	else
		return $key;
}

function translateValue($key,$val,$data=array())
{
	switch ($key) {
		case 'UPTIME':
		case 'WAIT':
			$min = floor(intval($val) / 60);
			$sec = intval($val) % 60;
			$val = "${min}mn${sec}s";
			if ($min > 180) {
				$hrs = floor($min / 60);
				$min = $min % 60;
				$val = "${hrs}h${min}mn";
			}
			break;
		case 'NAME':
			$val = $data['NAME'].'&nbsp;'.$data['VER'];
			break;
		case 'CPUFREQ':
		case 'FREQ':
		case 'MEMFREQ':
		case 'GPUF':
		case 'MEMF':
			$val = sprintf("%d MHz", $val);
			break;
		case 'POWER':
			$val = sprintf("%d W", round(floatval($val)/1000.0));
			break;
		case 'TS':
			$val = strftime("%H:%M:%S", (int) $val);
			break;
		case 'KHS':
		case 'NETKHS':
			$val = '<span class="bold">'.$val.'</span> kH/s';
			break;
		case 'KHW':
			$val = $val.' kH/W';
			break;
		case 'NAME':
		case 'POOL';
		case 'USER':
			// long fields
			$val = '<span class="elipsis">'.$val.'</span>';
			break;
	}
	return $val;
}

function filterPoolInfos($stats)
{
	$keys = array('USER','H','PING','DISCO');
	$data = array();
	$pool = array_pop($stats);
	// simplify URL to host only
	$data['POOL'] = $pool['URL'];
	if (strstr($pool['URL'],'://')) {
		$parts = explode(':', $pool['URL']);
		$data['POOL'] = substr($parts[1],2);
	}
	foreach ($pool as $key=>$val) {
		if (in_array($key, $keys))
			$data[$key] = $val;
	}
	return $data;
}

function displayData($data)
{
	$htm = '';
	$totals = array();
	foreach ($data as $name => $stats) {
		if (!isset($stats['summary']))
			continue;
		$htm .= '<table id="tb_'.$name.'" class="stats">'."\n";
		$htm .= '<tr><th class="machine" colspan="2">'.$name."</th></tr>\n";
		if (!empty($stats)) {
			$summary = (array) $stats['summary'];
			foreach ($summary as $key=>$val) {
				if (!empty($val) && !ignoreField($key))
					$htm .= '<tr><td class="key">'.translateField($key).'</td>'.
						'<td class="val">'.translateValue($key, $val, $summary)."</td></tr>\n";
			}
			if (isset($summary['KHS']))
				@ $totals[$summary['ALGO']] += floatval($summary['KHS']);

			if (isset($stats['pool']) && !empty($stats['pool']) ) {
				$pool = filterPoolInfos($stats['pool']);
				$htm .= '<tr><th class="gpu" colspan="2">POOL</th></tr>'."\n";
				foreach ($pool as $key=>$val) {
					if (!empty($val) && !ignoreField($key))
					$htm .= '<tr><td class="key">'.translateField($key).'</td>'.
						'<td class="val">'.translateValue($key, $val)."</td></tr>\n";
				}
			}

			foreach ($stats['threads'] as $g=>$gpu) {
				$card = isset($gpu['CARD']) ? $gpu['CARD'] : '';
				$htm .= '<tr><th class="gpu" colspan="2">'.$g." $card</th></tr>\n";
				foreach ($gpu as $key=>$val) {
					if (!empty($val) && !ignoreField($key))
					$htm .= '<tr><td class="key">'.translateField($key).'</td>'.
						'<td class="val">'.translateValue($key, $val)."</td></tr>\n";
				}
			}
		}
		$htm .= "</table>\n";
	}
	// totals
	if (!empty($totals)) {
		$htm .= '<div class="totals"><h2>Total Hash rate</h2>'."\n";
		foreach ($totals as $algo => $hashrate) {
			$htm .= '<li><span class="algo">'.$algo.":</span>$hashrate kH/s</li>\n";
		}
		$htm .= '</div>';
	}
	return $htm;
}

$data = getdataFromPeers();

?>
<html>
<head>
	<title>ccminer rig api sample</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<meta http-equiv="refresh" content="10">
<style type="text/css">
body {
	color:#cccccc; background:#1d1d1d; margin:30px 30px 0px 30px; padding:0px;
	font-size:.8em; font-family:Arial,Helvetica,sans-serif;
}
a { color:#aaaaaa; text-decoration: none; }
a:focus { outline-style:none; }
.clear { clear: both; }

div#page, div#header, div#footer {
	margin: auto;
	width: 950px;
	box-shadow: 0 5px 10px rgba(0, 0, 0, 0.15);
}
div#page {
	padding-top: 8px;
	background: #252525;
	min-height: 820px;
}
div#header {
	background: rgba(65, 65, 65, 0.85);
	height: 50px;
	margin-bottom: 24px;
	padding-left: 8px;
}
div#footer {
	background: rgba(25, 25, 25, 0.85);
	height: 0px;
	margin-bottom: 40px;
	text-align: center;
	color: #666666;
	text-shadow: rgba(0, 0, 0, 0.8) 0px 1px 0px;
}
#header h1 { padding: 12px; font-size: 20px; }
#footer p { margin: 12px 24px; }

table.stats { width: 280px; margin: 4px 16px; display: inline-block; vertical-align: top; }
th.machine { color: darkcyan; padding: 16px 0px 0px 0px; text-align: left; border-bottom: 1px solid gray; }
th.gpu { color: white; padding: 3px 3px; font: bolder; text-align: left; background: rgba(65, 65, 65, 0.85); }
td.key { width: 99px; max-width: 180px; }
td.val { width: 40px; max-width: 100px; color: white; }

div.totals { margin: 16px; padding-bottom: 16px; }
div.totals h2 { color: darkcyan; font-size: 16px; margin-bottom: 4px; }
div.totals li { list-style-type: none; font-size: 16px; margin-left: 4px; margin-bottom: 8px; }
li span.algo { display: inline-block; width: 100px; max-width: 180px; }

span.bold { color: #bb99aa; }
span.elipsis { display: inline-block; max-width: 130px; overflow: hidden; }
</style>
</head>
<body>
<div id="header">
<h1>ccminer monitoring API RIG sample</h1>
</div>

<div id="page">
<?=displayData($data)?>
</div>

<div id="footer">
<p>&copy; 2014-2015 <a href="http://github.com/tpruvot/ccminer">tpruvot@github</a></p>
</div>

</body>
</html>
