<html><body>
<h4>Rounded rectangles</h4>
Subset of 3000 (r=0..99, 30 examples each)
<p>
<table border=1 cellpadding=2 cellspacing=0>
<?php
//$imgs = glob("imgs-subset/*.png");
//foreach ($imgs as $img) {
for($r = 0; $r < 10; $r++ ) {
  echo "<tr><td>r = ${r}0</td>\n";
  for($i = 0; $i < 8; $i++ ) {
    $f = "mes-m${r}0-00$i.png";
    echo "<td><img src=\"imgs-subset/$f\"><br>$f</td>\n";
  }
  echo "</tr>\n";
}
?>
</table>
</body></html>
