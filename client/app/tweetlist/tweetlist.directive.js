(function() {
  'use strict';

  angular
    .module('bubbleApp.tweetlist')
    .directive('tweetList', tweetList);

  function tweetList() {
    return {
      restrict: 'E',
      controller: 'TweetListCtrl as vm',
      templateUrl: 'app/tweetlist/tweetlist.tmpl.html'
    };
  }
})();