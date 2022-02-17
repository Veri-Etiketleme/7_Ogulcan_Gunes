<?php

require "init.php";
$fcm_token=$_POST['fcm_token'];
$sql="insert into fcm_info3 values('".$fcm_token."');";
mysqli_query($con,$sql);
mysqli_close($con);
?>
